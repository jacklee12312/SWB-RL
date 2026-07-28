# Imported Deck Registry

`scripts.import_deck_qr` writes deterministic JSON manifests here after
decoding an official Shadowverse: Worlds Beyond deck QR image.

Every manifest preserves the official deck hash and ordered 40-card list,
resolves card metadata against `data/cards.sqlite3`, and records the exact-rule
coverage report hash. A manifest is exposed to the fixed-deck training and
evaluation selectors only when `validation.trainable` is `true` and
`validation.issues` is empty.

Untrainable imports may remain here for review, but they are deliberately not
silently admitted to RL training.
