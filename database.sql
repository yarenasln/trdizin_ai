DROP TABLE IF EXISTS article_subject;
DROP TABLE IF EXISTS article_text;
DROP TABLE IF EXISTS article;

CREATE TABLE article (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(255) UNIQUE,
    doi VARCHAR(300),
    publication_year INTEGER,
    publication_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE article_text (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES article(id) ON DELETE CASCADE,
    language VARCHAR(20) NOT NULL,
    title VARCHAR(1000) NOT NULL,
    abstract TEXT,
    keywords TEXT
);

CREATE TABLE article_subject (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES article(id) ON DELETE CASCADE,
    subject VARCHAR(500) NOT NULL
);