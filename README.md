# RNN-LM

Standard Recurrent Language Model.
	<p> Introduccion TO_DO</p>

## Instrucciones

#### Convert_text2dict.py

`python convert-text2dict.py ruta_del_fichero_de_entrada ruta_del_fichero_de_salida [--cutoff n] [--dict ruta_del_diccionario]`


#### Train.py
`THEANO_FLAGS=mode=FAST_RUN,floatX=float32,exception_verbosity=high,traceback.limit=0,optimizer=fast_compile python train.py [--resume ruta_del_modelo] [--force_train_all_wordemb] [--protoype nombre_del_prototipo]`

`THEANO_FLAGS=mode=FAST_RUN,floatX=float32,exception_verbosity=high,traceback.limit=0,optimizer=fast_compile python train.py --protoype prototype_largo`


#### Sample.py
`THEANO_FLAGS=mode=FAST_RUN,floatX=float32,exception_verbosity=high,traceback.limit=0,optimizer=fast_compile python sample.py ruta_del_modelo ruta_del_fichero_del_contexto ruta_del_fichero_de_salida [--ignore-unk] [--beam_search] [--n-samples n] [--n-turns k] [--normalize] [--verbose]`

#### Evaluate.py
`THEANO_FLAGS=mode=FAST_RUN,floatX=float32,exception_verbosity=high,traceback.limit=0,optimizer=fast_compile
 python evaluate.py ruta_del_modelo ruta_del_fichero_de_prueba_.pkl [--exclude-sos] [--plot-graphs]`  
 `[--exclude-stop-words] [--document-ids ruta_fichero_ids]`


  `THEANO_FLAGS=mode=FAST_RUN,floatX=float32,exception_verbosity=high,traceback.limit=0,optimizer=fast_compile python evaluate.py model/largo tests/tst.word.pkl`

## Cambios

###### 1. Cambios en ficheros
- **train.py:**
	1. Guardar el modelo
		- Cambio en linea: 223.
		- Añadida linea: save(model).
		- Guarda el modelo.
	2. Prototipo de estado a usar por defecto.
		- Cambio en linea: 230.
		- Sustituye el prototipo a usar por defecto.
		- Prototype_train (creado por mi) en lugar de prototype_web (que no estaba implementado).


- **recurrent_lm.py:**
	- Cambio en linea: 389.
	- variable: self.dictionary.
	- por el nombre de los ficheros de diccionario creados por convert_text2dict.


- **state.py:**
	- creado nuevo protoype. (protoype_train)


- **evaluate.py:**
	- quizas cambiar en linea 71 el prototype (protoype_train)
	- En linea 351, cambia el nombre de la ruta para guardar la imagen

###### 2. Ficheros y directorios creados
- **/Data:** Directorio con ficheros de datos.
	- **context_input_sample.txt:**		Fichero entrada para sample.py.
	- **context_output_sample.txt:**	Fichero salida de sample.py.
	- **test_dataset.txt:**				Fichero conjunto de testeo.
	- **training_dataset.txt:**			Fichero conjunto de entrenamiento.
	- **validation_dataset.txt:**		Fichero conjunto de validacion.
	- **PKL:** 							Ficheros creados por convert_convert_text2dict.
		+ **test.dict.pkl** Diccionario conjunto testeo.
		+ **train.dict.pkl** Diccionario conjunto entrenamiento.
		+ **validation.dict.pkl** Diccionario conjunto validacion.
		+ **test.word.pkl** Conjunto palabras conjunto testeo.
		+ **train.word.pkl** Conjunto palabras conjunto entrenamiento.
		+ **validation.word.pkl** Conjunto palabras conjunto validacion.


- **/model:** Ficheros del modelo, donde RUN_ID es sustituido por un numero.
	- **RUN_ID_lm_train_model.npz:** Modelo.
	- **RUN_ID_lm_train_state.pkl:** Estado.
	- **RUN_ID_lm_train_timing.npz:** Tiempo.
