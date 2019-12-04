# RNN-LM

Standard Recurrent Language Model.

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
