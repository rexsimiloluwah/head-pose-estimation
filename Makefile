install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint app.py

start-app: 
	streamlit run app.py

run-docker: 
	docker run -p 8501:8501 ${IMAGE}