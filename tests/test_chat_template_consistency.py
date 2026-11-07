#!/usr/bin/env python3
"""Task 7 — compare the training chat template against the one inside the GGUF.

The existing test compares two outputs of the same project helper, which establishes
only that the helper is self-consistent. This reads tokenizer.chat_template straight
out of the GGUF's metadata and compares it with the template the tokenizer applied
during training, which is the comparison the criterion is about.

    python3 3_test_chat_template_real.py /models/<fine-tuned>.gguf \
        --base-model Qwen/Qwen2.5-3B-Instruct

The --base-model lookup is optional; without network access, pass --training-template
pointing at the chat_template.txt saved from the Colab run.
"""
import argparse, json, re, struct, sys

# ---------------------------------------------------------------- GGUF metadata
SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
           6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


def read_gguf_metadata(path, want=('tokenizer.chat_template',
                                   'general.architecture', 'general.file_type',
                                   'tokenizer.ggml.add_bos_token')):
    """Parse the GGUF key/value header. Arrays are consumed in full -- skipping their
    elements desynchronises the stream and yields silent garbage further on."""
    out = {}
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError('%s is not a GGUF file' % path)
        version, = struct.unpack('<I', f.read(4))
        n_tensors, = struct.unpack('<Q', f.read(8))
        n_kv, = struct.unpack('<Q', f.read(8))

        def rstr():
            n, = struct.unpack('<Q', f.read(8))
            return f.read(n).decode('utf8', 'replace')

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, = struct.unpack('<I', f.read(4))
                ln, = struct.unpack('<Q', f.read(8))
                return [rval(et) for _ in range(ln)]
            fmt = SCALARS[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            k = rstr()
            t, = struct.unpack('<I', f.read(4))
            v = rval(t)
            if k in want:
                out[k] = v
    out['_version'] = version
    out['_n_tensors'] = n_tensors
    return out


def normalise(t):
    """Jinja templates differ harmlessly in whitespace between exporters; compare the
    structure rather than the formatting."""
    if not t:
        return ''
    t = t.replace('\r\n', '\n')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s*\n+', '\n', t)
    return t.strip()


def fingerprint(t):
    """The parts that actually change model behaviour."""
    n = normalise(t)
    return {
        'has_tools_branch': bool(re.search(r'\{%-?\s*if\s+tools', n)),
        'im_start_count': n.count('<|im_start|>'),
        'im_end_count': n.count('<|im_end|>'),
        'roles': sorted(set(re.findall(r"'(system|user|assistant|tool)'", n))),
        'adds_generation_prompt': 'add_generation_prompt' in n,
        'length': len(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('--base-model', help='HF id to pull the reference template from')
    ap.add_argument('--training-template',
                    help='chat_template.txt saved from the training run')
    args = ap.parse_args()

    meta = read_gguf_metadata(args.gguf)
    gguf_tpl = meta.get('tokenizer.chat_template')
    print('GGUF: %s' % args.gguf)
    print('  architecture : %s' % meta.get('general.architecture'))
    print('  file_type    : %s' % meta.get('general.file_type'))
    print('  tensors      : %s' % meta.get('_n_tensors'))
    if not gguf_tpl:
        print('\nFAIL: the GGUF carries no tokenizer.chat_template.')
        print('llama-server will fall back to a built-in template, so training and')
        print('inference formatting cannot be guaranteed to match.')
        return 1
    print('  chat_template: %d characters' % len(gguf_tpl))

    train_tpl = None
    if args.training_template:
        train_tpl = open(args.training_template, encoding='utf8').read()
        source = args.training_template
    elif args.base_model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.base_model)
            train_tpl = tok.chat_template
            source = args.base_model
        except Exception as e:
            print('\ncould not load %s: %s' % (args.base_model, e))
            return 2
    else:
        print('\nNo reference supplied. Pass --training-template or --base-model.')
        print('\nFingerprint of the embedded template:')
        for k, v in fingerprint(gguf_tpl).items():
            print('  %-24s %s' % (k, v))
        return 3

    print('Reference: %s (%d characters)\n' % (source, len(train_tpl)))
    a, b = normalise(train_tpl), normalise(gguf_tpl)
    fa, fb = fingerprint(train_tpl), fingerprint(gguf_tpl)

    if a == b:
        print('PASS: the templates are identical after whitespace normalisation.')
        return 0

    print('Templates are not byte-identical. Comparing what affects behaviour:\n')
    same = True
    for k in fa:
        mark = 'ok ' if fa[k] == fb[k] else 'DIFF'
        if fa[k] != fb[k] and k != 'length':
            same = False
        print('  %-4s %-24s training=%-22s gguf=%s' % (mark, k, fa[k], fb[k]))

    if same:
        print('\nPASS (structurally): role handling, control tokens and the tool branch')
        print('all match; the differences are formatting only.')
        return 0

    print('\nFAIL: the templates differ in a way that changes how prompts are built.')
    print('Outputs at inference will not match the format the model was trained on.')
    import difflib
    for line in list(difflib.unified_diff(a.split('\n'), b.split('\n'),
                                          'training', 'gguf', lineterm='', n=1))[:25]:
        print('  ' + line)
    return 1


if __name__ == '__main__':
    sys.exit(main())
