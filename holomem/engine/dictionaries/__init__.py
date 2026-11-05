"""Language data the matcher reads, each built by a script in tools/ and licensed as its source (texts in LICENSES/):

- cjk_fold.py — Traditional -> Simplified Chinese characters, from OpenCC (Apache-2.0); tools: none (generated once).
- ru_derivation/ — Russian words of one root, from DerivBase.Ru in UDer 1.1 (Apache-2.0); build_ru_derivation.py.
- en_derivation/ — English words of one root, from Open English Wordnet 2025 (https://en-word.net/,
  https://github.com/globalwordnet/english-wordnet; (c) The Open English Wordnet Team, CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/; on Princeton WordNet 3.1); build_en_derivation.py.
- ja_variants/ — Japanese spelling variants, from SudachiDict's synonym source (Apache-2.0); build_ja_variants.py.
- zh_words/ — Chinese words and their frequencies, from jieba's dictionary (MIT); build_zh_words.py.
- de_compounds/, de_es_forms/ — German compounds, German and Spanish verb forms, from Wikidata Lexemes
  (CC0); build_de_es_lexicon.py.

Each table is a folder of plain UTF-8 parts under 256 KiB (01.tsv, 02.tsv, …: "# part i/n" first, "# end i/n" last;
text._read_table), which the Claude plugin directory reads as text. Each loads on the first word of its language; a
copy without a table works without that language's extra."""
