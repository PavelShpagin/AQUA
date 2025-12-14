#!/usr/bin/env python3
"""
aligned_text.py

Generate inline aligned text from a source and target sentence using spaCy and ERRANT.

Usage:
    pip install spacy errant
    python -m spacy download en_core_web_sm
    python aligned_text.py "I like turtles" "I like turtles and pandas"

This will output:
    I like {turtles=>turtles} {=>and pandas}
"""

import argparse
import spacy
import errant
import sys
import os
from spacy.tokens import Doc
from spacy.lang.en import English

# Add project root and ua-gec module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ua-gec', 'python'))
from ua_gec.annotated_text import AnnotatedText


def custom_tokenize_spaces(text):
    """
    Tokenize text treating each space as a single token.
    This prevents spaCy from grouping multiple spaces into one token.
    """
    tokens = []
    current_word = ''
    
    for char in text:
        if char == ' ':
            if current_word:
                tokens.append(current_word)
                current_word = ''
            tokens.append(' ')
        else:
            current_word += char
    
    if current_word:
        tokens.append(current_word)
    
    return tokens


def create_custom_doc(nlp, text):
    """
    Create a spaCy Doc with custom tokenization that treats each space as a single token.
    """
    tokens = custom_tokenize_spaces(text)
    
    # Create spaCy Doc manually with proper spacing
    words = []
    spaces = []
    
    for i, token in enumerate(tokens):
        words.append(token)
        # Space tokens should not have trailing space, non-space tokens should
        if token == ' ':
            spaces.append(False)  # Space tokens don't have trailing space
        else:
            # Check if next token is a space or if this is the last token
            if i + 1 < len(tokens) and tokens[i + 1] == ' ':
                spaces.append(False)  # No trailing space if next token is space
            else:
                spaces.append(i < len(tokens) - 1)  # Normal spacing logic
    
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    
    # Run the pipeline to get POS tags and other attributes ERRANT needs
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    
    return doc


def make_inline_alignment(src_doc, tgt_doc, edits, src_text, tgt_text):
    """
    Simple alignment using spaCy's native text_with_ws - handles all spacing automatically.
    """
    parts = []
    idx = 0
    
    for edit in edits:
        # Add unchanged tokens
        parts.append(''.join([t.text_with_ws for t in src_doc[idx:edit.o_start]]))
        # Add edit
        o_seg = ''.join([t.text for t in src_doc[edit.o_start:edit.o_end]])
        c_seg = ''.join([t.text for t in tgt_doc[edit.c_start:edit.c_end]])
        parts.append(f"{{{o_seg}=>{c_seg}}}")
        idx = edit.o_end
    
    # Add remaining tokens
    parts.append(''.join([t.text_with_ws for t in src_doc[idx:]]))
    
    return ''.join(parts).rstrip()


def verify_alignment(aligned_text, target_text):
    """
    Verify that the aligned text produces the correct target text when corrections are applied.
    Returns True if alignment is correct, False otherwise.
    """
    try:
        corrected = AnnotatedText(aligned_text).get_corrected_text()
        
        # Strict comparison - no cheating with normalization
        return corrected.strip() == target_text.strip()
    except Exception as e:
        print(f"Error during alignment verification: {e}")
        return False


def main():
    # Parse command-line arguments

    default_src = "Coding I like"
    default_tgt = "I like coding"
    parser = argparse.ArgumentParser(
        description="Generate inline aligned text from source and target sentences (ERRANT, space-preserving)."
    )
    parser.add_argument("src", nargs='?', default=default_src, help="Source sentence to align.")
    parser.add_argument("tgt", nargs='?', default=default_tgt, help="Target (corrected) sentence to align.")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model to load (default: en_core_web_sm)")
    parser.add_argument("--engine", choices=["local", "utils"], default="utils", help="Alignment engine: local (this script) or utils.errant_align.get_alignment_for_language")
    parser.add_argument("--merge", dest="merge", action="store_true", help="Merge adjacent edits in output")
    parser.add_argument("--no-merge", dest="merge", action="store_false", help="Do not merge adjacent edits in output")
    parser.set_defaults(merge=True)
    parser.add_argument("--preserve-spaces", dest="preserve_spaces", action="store_true", help="Use space-preserving tokenization (utils engine)")
    parser.add_argument("--no-preserve-spaces", dest="preserve_spaces", action="store_false", help="Disable space-preserving tokenization (utils engine)")
    parser.set_defaults(preserve_spaces=True)
    parser.add_argument("--show-produced", action="store_true", help="Show text produced by applying alignment to source")
    parser.add_argument("--compare-dmp", action="store_true", help="Also compute a diff-match-patch alignment (requires python-diff-match-patch)")
    args = parser.parse_args()

    # Load spaCy and ERRANT
    nlp = spacy.load(args.model)
    annotator = errant.load("en", nlp)

    if args.engine == "local":
        # Parse texts into Docs using custom tokenization (preserving all whitespace)
        src_doc = create_custom_doc(nlp, args.src)
        tgt_doc = create_custom_doc(nlp, args.tgt)

        # Verify reconstruction
        src_reconstructed = ''.join([t.text_with_ws for t in src_doc])
        tgt_reconstructed = ''.join([t.text_with_ws for t in tgt_doc])
        print(f"Source reconstruction: {repr(src_reconstructed)} == {repr(args.src)} ? {src_reconstructed == args.src}")
        print(f"Target reconstruction: {repr(tgt_reconstructed)} == {repr(args.tgt)} ? {tgt_reconstructed == args.tgt}")
        print()

        # Annotate to get Edit objects
        edits = annotator.annotate(src_doc, tgt_doc)

        # Build aligned string
        aligned = make_inline_alignment(src_doc, tgt_doc, edits, args.src, args.tgt)

        # Optional post-merge using utils.merge_alignment for parity with pipeline
        if args.merge:
            try:
                from utils.errant_align import merge_alignment as _merge
                aligned = _merge(aligned)
            except Exception as e:
                print(f"Warn: merge skipped ({e})")
    else:
        # Use the production helper with configurable merging and spacing
        try:
            from utils.errant_align import get_alignment_for_language
        except Exception as e:
            print(f"Failed to import utils.errant_align: {e}")
            sys.exit(1)
        aligned = get_alignment_for_language(
            src_text=args.src,
            tgt_text=args.tgt,
            language='en',
            nlp=nlp,
            annotator=annotator,
            preserve_spaces=args.preserve_spaces,
            merge=args.merge,
        )

    print("Aligned text:")
    print(aligned)

    # Verify alignment correctness
    print("\nAlignment verification:")
    is_correct = verify_alignment(aligned, args.tgt)
    print("✓ Alignment is CORRECT - produces expected target text" if is_correct else "✗ Alignment is INCORRECT - does not match target text")

    if args.show_produced:
        try:
            produced = AnnotatedText(aligned).get_corrected_text()
            print(f"Produced corrected: {produced}")
        except Exception as e:
            print(f"Could not extract corrected text: {e}")

    if args.compare_dmp:
        try:
            import diff_match_patch as dmp_module
            dmp = dmp_module.diff_match_patch()
            diffs = dmp.diff_main(args.src, args.tgt)
            dmp.diff_cleanupSemantic(diffs)
            # Render a very simple DMP-based inline diff for comparison
            src_out = []
            tgt_out = []
            for op, data in diffs:
                if op == 0:  # equal
                    src_out.append(data)
                    tgt_out.append(data)
                elif op == -1:  # delete
                    src_out.append(f"{{{data}=>}}")
                elif op == 1:  # insert
                    tgt_out.append(f"{{=>{data}}}")
            print("\nDMP diff (source-marked deletes + target-marked inserts):")
            print(''.join(src_out).strip())
            print(''.join(tgt_out).strip())
        except Exception as e:
            print(f"diff-match-patch not available or failed: {e}")


if __name__ == "__main__":
    main()
