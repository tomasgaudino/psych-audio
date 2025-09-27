ENV=psych-audio
PY=python

.PHONY: create-env install activate run spacy-model clean uninstall

create-env:
	conda env create -f environment.yml

install: spacy-model
	@echo "Entorno listo."

spacy-model:
	conda activate $(ENV) && $(PY) -m pip install -U pip setuptools wheel && \
	$(PY) -m spacy download es_core_news_md

activate:
	@echo "Run: conda activate $(ENV)"

run:
	conda activate $(ENV) && \
	$(PY) whatsapp_psych_session.py \
	  --audio "sample.ogg" \
	  --output "salida.md" \
	  --denoise \
	  --whisper-model "large-v3" \
	  --hf-token-env "HUGGINGFACE_TOKEN"

clean:
	rm -f salida.md

uninstall:
	conda remove --name $(ENV) --all -y
