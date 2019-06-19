00: doc := 73415599579f2588080e.md
00:
	python ../tools/parse.py ${doc} | python ../tools/topycode.py > all.py
	nbreversible all.py > all.ipynb

notebook:
	jupyter-notebook --ip 127.0.0.1 --no-browser
