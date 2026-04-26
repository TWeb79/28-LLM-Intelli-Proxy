.PHONY: build up down logs test scan

build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -q

scan:
	# placeholder for static analysis (flake8, bandit, etc.)
	echo "Run linters here"
