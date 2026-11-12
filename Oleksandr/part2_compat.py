"""Python 3.13 compatibility for the frozen-split guard, without editing part2_data.py.

Python 3.13 stopped printing empty AST fields in ast.dump, so part2_data.grouping_hash
returns a new value and read_split rejects the unchanged split ("Grouping functions
changed"). Hashing the pre-3.13 form (show_empty=True) restores the recorded value;
Part 2 uses the same fix on Drive (word_representations_v1/grouping_runtime.json,
Python 3.13.16). On older Pythons this is a no-op.
"""
import ast
import inspect

import part2_data


def pin_grouping_hash():
    """Make part2_data.read_split hash the grouping code as Python 3.12 did; returns True if patched."""
    if 'show_empty' not in inspect.signature(ast.dump).parameters:
        return False
    functions = [part2_data.fingerprint, part2_data.grouping_text, part2_data.normalized_words,
                 part2_data.separator_signature, part2_data.text_group]

    def grouping_hash():
        return part2_data.fingerprint([ast.dump(ast.parse(inspect.getsource(f)), include_attributes=False,
                                                show_empty=True) for f in functions])
    part2_data.grouping_hash = grouping_hash
    return True
