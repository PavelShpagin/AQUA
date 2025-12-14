#!/usr/bin/env python3
"""
utils/diff_extension.py
=======================

Thin, clean wrapper around google-diff-match-patch for word-level diffs and
lightweight edit metrics. Intended for prompt/context construction (not for
alignment labels).

Exports:
- diff_by_words(text1, text2, cleanup_semantic=False) -> List[(op, text)]
- char_level_levenshtein_distance(s1, s2) -> int
"""

from __future__ import annotations

from typing import Iterator, Tuple, List
import diff_match_patch as dmp_module


def is_finished_sentence(s: str, pos: int | None = None) -> bool:
    if pos is not None:
        s = s[: pos + 1]

    s = s.rstrip()
    if not s:
        return False

    if s.endswith("''"):
        s = s[:-2].rstrip()
    elif s[-1] in ('"', "'", ")", "]", "»"):
        s = s[:-1].rstrip()

    return bool(s) and s[-1] in "….?!"


# Character classes
whitespace_and_newlines = {
    "\u0020", "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002", "\u2003",
    "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
    "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\u000a", "\u000b",
    "\u0009", "\u000c", "\u000d", "\u0085",
}

alternative_dash = {"–", "—", "‒", "―", "−", "﹣", "－"}
dual_quotation = {'"', "'", "‛", "`", "‟"}
opening_quotation = {'"', "«", "‹", "“", "„", "’", "‘", "‚"}
closing_quotation = {'"', "”", "»", "›", "’"}
separator_punctuation = {",", "?", "!", ":", ";", "…", "•"}
OBJECT_REPLACEMENT_CHAR = "\ufffc"
special_characters = {OBJECT_REPLACEMENT_CHAR}

word_delimiters = (
    whitespace_and_newlines
    | alternative_dash
    | dual_quotation
    | opening_quotation
    | closing_quotation
    | separator_punctuation
    | special_characters
)


def is_word_delimiter(ch: str) -> bool:
    return ch in word_delimiters


def _tokenize_text(text: str) -> Iterator[Tuple[str, int, int]]:
    lineStart = 0
    lineEnd = -1

    def findWordEndPosition() -> int:
        length = len(text)
        index = lineStart
        while index < length:
            ch = text[index]
            if is_word_delimiter(ch) or (ch == "." and is_finished_sentence(text, index)):
                if (
                    text[index] in {"'", "’"}
                    and index != 0
                    and text[index - 1] not in whitespace_and_newlines
                    and index + 1 < length
                    and text[index + 1] not in whitespace_and_newlines
                ):
                    index += 1
                elif index - lineStart >= 1:
                    return index - 1
                else:
                    return index
            index += 1
        return -1

    while lineEnd < len(text) - 1:
        lineEnd = findWordEndPosition()
        if lineEnd == -1:
            lineEnd = len(text) - 1
        line = text[lineStart : lineEnd + 1]
        yield line, lineStart, lineEnd + 1
        lineStart = lineEnd + 1


def _diff_linesToWords(text1: str, text2: str) -> tuple[str, str, List[str]]:
    lineArray: List[str] = [""]
    lineHash: dict[str, int] = {}

    def diff_linesToCharsMunge(text: str, maxLines: int) -> str:
        chars: List[str] = []
        for line, lineStart, _lineEnd in _tokenize_text(text):
            if line in lineHash:
                chars.append(chr(lineHash[line]))
            else:
                if len(lineArray) == maxLines:
                    line = text[lineStart:]
                lineArray.append(line)
                lineHash[line] = len(lineArray) - 1
                chars.append(chr(len(lineArray) - 1))
                if len(lineArray) == maxLines:
                    break
        return "".join(chars)

    # Allocate 2/3rds of the space for text1, the rest for text2.
    chars1 = diff_linesToCharsMunge(text1, 40000)
    chars2 = diff_linesToCharsMunge(text2, 65535)
    return (chars1, chars2, lineArray)


def diff_by_words(original_text: str, new_text: str, cleanup_semantic: bool = False):
    if original_text == new_text:
        return [(0, original_text)]
    dmp = dmp_module.diff_match_patch()
    words1, words2, words_array = _diff_linesToWords(original_text, new_text)
    diffs = dmp.diff_main(words1, words2, False)
    if cleanup_semantic:
        dmp.diff_cleanupSemantic(diffs)
    dmp.diff_charsToLines(diffs, words_array)
    return diffs


def char_level_levenshtein_distance(s1: str, s2: str) -> int:
    dmp = dmp_module.diff_match_patch()
    return dmp.diff_levenshtein(dmp.diff_main(s1, s2, False))






