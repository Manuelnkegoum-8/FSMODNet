# ==========================================================
# Modified from mmcv
# ==========================================================
import os, sys
import os.path as osp
import ast
import tempfile
import shutil
from importlib import import_module

from argparse import Action

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

import platform
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text', 'get', 'dump', 'merge_from_dict']


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class _BaseProxy:
    """
    Proxy object injected as ``_base_`` when a child config is executed.

    Enables multi-base attribute access in two ways:

        # Search all bases in order (first match wins)
        mean = _base_.mean

        # Access a specific base by index
        mean = _base_[0].mean
        lr   = _base_[1].optimizer.lr

    The proxy wraps each base as a :class:`ConfigDict`, so nested attribute
    access (``_base_[0].model.backbone``) works out of the box.
    """

    def __init__(self, bases: list):
        # Store as ConfigDicts so nested attr access works on each element
        object.__setattr__(self, '_bases', [
            b if isinstance(b, ConfigDict) else ConfigDict(b)
            for b in bases
        ])

    def __getattr__(self, name):
        """Search all bases in order; return first match."""
        for base in object.__getattribute__(self, '_bases'):
            try:
                return base[name]
            except KeyError:
                continue
        raise AttributeError(
            f"None of the base configs contain attribute '{name}'. "
            f"Available keys per base: "
            + str([list(b.keys()) for b in object.__getattribute__(self, '_bases')])
        )

    def __getitem__(self, idx):
        """Access a specific base config by index."""
        return object.__getattribute__(self, '_bases')[idx]

    def __len__(self):
        return len(object.__getattribute__(self, '_bases'))

    def __repr__(self):
        bases = object.__getattribute__(self, '_bases')
        return f'_BaseProxy({len(bases)} base(s))'


class _StripBaseAssignment(ast.NodeTransformer):
    """AST transformer that removes ``_base_ = ...`` top-level assignments.

    This is required so that when a child config is ``exec``'d, the
    ``_base_ = [...]`` line does not overwrite the :class:`_BaseProxy` we
    injected into the exec namespace before running the file.
    """

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == BASE_KEY:
                return None   # drop this statement entirely
        return node

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == BASE_KEY:
            return None
        return node


class SLConfig(object):
    """
    Config files.
    Supports .py / .yml / .yaml / .json.

    Multi-base inheritance
    ----------------------
    When ``_base_`` is a list, child configs can access base attributes via
    the injected :class:`_BaseProxy`::

        _base_ = ['detection/backbone.py', 'detection/head.py']

        # First-match attribute lookup across all bases:
        normalizer = dict(type='BN', mean=_base_.mean, std=_base_.std)

        # Index into a specific base:
        lr = _base_[1].optimizer.lr

    ref: mmcv.utils.config
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename) as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError:
            raise SyntaxError(f'There are syntax errors in config file {filename}')

    @staticmethod
    def _extract_base_files(filename):
        """
        Extract the value of ``_base_`` from *filename* using the AST,
        without importing the file.  Returns a (possibly empty) list of
        relative path strings.
        """
        with open(filename) as f:
            content = f.read()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == BASE_KEY:
                            val = ast.literal_eval(node.value)
                            return [val] if isinstance(val, str) else list(val)
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # core loader
    # ------------------------------------------------------------------

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)

        base_cfg_dicts_list = []   # list[dict]  – one per base file
        cfg_text_list = []         # list[str]

        if filename.lower().endswith('.py'):
            SLConfig._validate_py_syntax(filename)
            cfg_dir = osp.dirname(filename)

            # ── Step 1: pre-load every base config ──────────────────────
            base_filenames = SLConfig._extract_base_files(filename)
            for bf in base_filenames:
                _cfg_dict, _cfg_text = SLConfig._file2dict(osp.join(cfg_dir, bf))
                base_cfg_dicts_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            # ── Step 2: exec child config with _BaseProxy injected ───────
            # We use exec() so we can pre-populate the namespace with the
            # proxy.  Critically, we also strip the ``_base_ = [...]``
            # assignment from the AST *before* exec so it cannot overwrite
            # the proxy we just injected.
            exec_namespace = {'__file__': filename}
            if base_cfg_dicts_list:
                exec_namespace[BASE_KEY] = _BaseProxy(base_cfg_dicts_list)

            with open(filename) as f:
                code_content = f.read()

            # Parse → strip _base_ assignment → recompile → exec
            tree = ast.parse(code_content, filename)
            tree = _StripBaseAssignment().visit(tree)
            ast.fix_missing_locations(tree)
            exec(compile(tree, filename, 'exec'), exec_namespace)

            cfg_dict = {
                k: v for k, v in exec_namespace.items()
                if not k.startswith('__')
            }
            # Drop the proxy object – it is not part of the config data
            cfg_dict.pop(BASE_KEY, None)

        elif filename.lower().endswith(('.yml', '.yaml', '.json')):
            from .slio import slload
            cfg_dict = slload(filename)

            # YAML/JSON: handle _base_ the old way (no proxy needed there)
            if BASE_KEY in cfg_dict:
                cfg_dir = osp.dirname(filename)
                raw = cfg_dict.pop(BASE_KEY)
                base_filenames = [raw] if isinstance(raw, str) else raw
                for bf in base_filenames:
                    _cfg_dict, _cfg_text = SLConfig._file2dict(osp.join(cfg_dir, bf))
                    base_cfg_dicts_list.append(_cfg_dict)
                    cfg_text_list.append(_cfg_text)
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        # ── Step 3: collect cfg_text ─────────────────────────────────────
        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        # ── Step 4: merge base configs (same logic as before) ────────────
        if base_cfg_dicts_list:
            base_cfg_dict = dict()
            for c in base_cfg_dicts_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Conflicting keys: '
                                   f'{base_cfg_dict.keys() & c.keys()}')
                base_cfg_dict.update(c)

            cfg_dict = SLConfig._merge_a_into_b(cfg_dict, base_cfg_dict)

        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    # ------------------------------------------------------------------
    # merge helper (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_a_into_b(a, b):
        """Merge dict ``a`` into dict ``b`` (non-inplace). Values in ``a``
        overwrite ``b``."""
        if not isinstance(a, dict):
            return a

        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict) and not isinstance(b[k], list):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base config')
                b[k] = SLConfig._merge_a_into_b(v, b[k])
            elif isinstance(b, list):
                try:
                    _ = int(k)
                except Exception:
                    raise TypeError(
                        f'b is a list, index {k} should be an int '
                        f'when input but {type(k)}')
                b[int(k)] = SLConfig._merge_a_into_b(v, b[int(k)])
            else:
                b[k] = v
        return b

    # ------------------------------------------------------------------
    # public API (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = SLConfig._file2dict(filename)
        return SLConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(SLConfig, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(SLConfig, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(SLConfig, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            return _indent(attr_str, indent)

        def _format_list(k, v, use_mapping=False):
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                return _indent(attr_str, indent) + ']'
            return _format_basic_types(k, v, use_mapping)

        def _contain_invalid_identifier(dict_str):
            return any(not str(k).isidentifier() for k in dict_str)

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []
            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end
                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)
        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, file=None):
        if file is None:
            return self.pretty_text
        with open(file, 'w') as f:
            f.write(self.pretty_text)

    def merge_from_dict(self, options):
        """Merge list into cfg_dict.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp': True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            d[key_list[-1]] = v

        cfg_dict = super(SLConfig, self).__getattribute__('_cfg_dict')
        super(SLConfig, self).__setattr__(
            '_cfg_dict', SLConfig._merge_a_into_b(option_cfg_dict, cfg_dict))

    def __setstate__(self, state):
        self.__init__(state)

    def copy(self):
        return SLConfig(self._cfg_dict.copy())

    def deepcopy(self):
        return SLConfig(self._cfg_dict.deepcopy())


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can be
    passed as comma-separated values: KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return val.lower() == 'true'
        if val.lower() in ['none', 'null']:
            return None
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)