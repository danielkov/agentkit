.PHONY: fmt fmt-book fmt-code

fmt: fmt-code fmt-book ## Format source code and book

fmt-code: ## Format Rust source code
	cargo fmt --all

fmt-book: ## Format book markdown files
	npx prettier --write book/src/**/*.md book/src/*.md
