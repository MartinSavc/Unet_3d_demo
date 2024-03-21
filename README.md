# Unet 3D primer

### requirements.txt
Knjižnice, ki jih potrebujemo. Trenutno brez specifičnih verzij. Pripravimo si lahko virtualno okolje:
```
$ python -m venv env
$ source env/bin/activate
```
In nato namestimo knjižnice z pip ukazom:
```
$ pip install -r requirements.txt
```

### model.py
Pripravljen UNet3D model. 
### loader.py
Primer za nalaganje podatkov/dataset. Rahlo kompleksnejši, trenutno generira naključne podatke.
### train.py
Učenje modela. Če poženemo to skripto, bo poskusala naučiti model na naključnih podatkih iz ```loader.py```. Shranila bo model na zadnjem koraku učenja.
### predict.py
Primer nalaganja in uporabe naučenega modela.
