PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS card_sets (
    id INTEGER PRIMARY KEY,
    is_collectible INTEGER NOT NULL DEFAULT 1 CHECK (is_collectible IN (0, 1))
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS rarities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS card_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS cards (
    card_id INTEGER PRIMARY KEY,
    base_card_id INTEGER NOT NULL,
    card_set_id INTEGER NOT NULL REFERENCES card_sets(id),
    class_id INTEGER NOT NULL REFERENCES classes(id),
    rarity_id INTEGER NOT NULL REFERENCES rarities(id),
    type_id INTEGER NOT NULL REFERENCES card_types(id),
    cost INTEGER NOT NULL CHECK (cost >= 0),
    attack INTEGER,
    life INTEGER,
    is_evolution INTEGER NOT NULL CHECK (is_evolution IN (0, 1)),
    evolves_to INTEGER,
    tribe_id INTEGER NOT NULL DEFAULT 0,
    tribe_name TEXT NOT NULL DEFAULT '',
    name_pinyin TEXT NOT NULL DEFAULT '',
    name_romaji TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL,
    CHECK (attack IS NULL OR attack >= 0),
    CHECK (life IS NULL OR life >= 0)
);

CREATE TABLE IF NOT EXISTS card_names (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    language TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (card_id, language)
);

CREATE TABLE IF NOT EXISTS skills (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    skill_id INTEGER NOT NULL,
    type INTEGER NOT NULL,
    subtype INTEGER NOT NULL,
    PRIMARY KEY (card_id, position),
    UNIQUE (skill_id)
);

CREATE TABLE IF NOT EXISTS skill_texts (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text_key TEXT NOT NULL,
    text TEXT NOT NULL,
    text_chs TEXT NOT NULL DEFAULT '',
    text_cht TEXT NOT NULL DEFAULT '',
    text_eng TEXT NOT NULL DEFAULT '',
    text_jpn TEXT NOT NULL DEFAULT '',
    text_kor TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (card_id, position)
);

CREATE TABLE IF NOT EXISTS card_localizations (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    language TEXT NOT NULL,
    class_name TEXT NOT NULL DEFAULT '',
    rarity_name TEXT NOT NULL DEFAULT '',
    type_name TEXT NOT NULL DEFAULT '',
    tribe_name TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (card_id, language)
);

CREATE TABLE IF NOT EXISTS flavor_texts (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text_key TEXT NOT NULL,
    text_chs TEXT NOT NULL DEFAULT '',
    text_cht TEXT NOT NULL DEFAULT '',
    text_eng TEXT NOT NULL DEFAULT '',
    text_jpn TEXT NOT NULL DEFAULT '',
    text_kor TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (card_id, position)
);

CREATE TABLE IF NOT EXISTS alt_modes (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    mode_type TEXT NOT NULL,
    cost INTEGER,
    text_chs TEXT NOT NULL DEFAULT '',
    text_cht TEXT NOT NULL DEFAULT '',
    text_eng TEXT NOT NULL DEFAULT '',
    text_jpn TEXT NOT NULL DEFAULT '',
    text_kor TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (card_id, position)
);

CREATE TABLE IF NOT EXISTS card_references (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    referenced_card_id INTEGER,
    referenced_name TEXT NOT NULL,
    PRIMARY KEY (card_id, position)
);

CREATE TABLE IF NOT EXISTS card_extra_data (
    card_id INTEGER PRIMARY KEY REFERENCES cards(card_id) ON DELETE CASCADE,
    skin_names TEXT NOT NULL DEFAULT '{}',
    voices TEXT NOT NULL DEFAULT '{}',
    voice_variants TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS textures (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    variant TEXT NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (card_id, variant)
);

CREATE TABLE IF NOT EXISTS rule_support (
    card_id INTEGER PRIMARY KEY REFERENCES cards(card_id) ON DELETE CASCADE,
    support_level TEXT NOT NULL CHECK (
        support_level IN ('basic', 'keyword', 'unsupported')
    ),
    keywords TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS abilities (
    keyword TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('implemented', 'partial', 'placeholder')
    ),
    events TEXT NOT NULL DEFAULT '[]',
    aliases TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS card_abilities (
    card_id INTEGER NOT NULL REFERENCES cards(card_id) ON DELETE CASCADE,
    ability_keyword TEXT NOT NULL REFERENCES abilities(keyword),
    raw_keyword TEXT NOT NULL,
    PRIMARY KEY (card_id, ability_keyword)
);

CREATE TABLE IF NOT EXISTS source_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    card_count INTEGER NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cards_class ON cards(class_id);
CREATE INDEX IF NOT EXISTS idx_cards_type ON cards(type_id);
CREATE INDEX IF NOT EXISTS idx_cards_set ON cards(card_set_id);
CREATE INDEX IF NOT EXISTS idx_cards_cost ON cards(cost);
CREATE INDEX IF NOT EXISTS idx_card_names_name ON card_names(name);
CREATE INDEX IF NOT EXISTS idx_card_abilities_keyword
    ON card_abilities(ability_keyword);
