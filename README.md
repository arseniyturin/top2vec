# Top2Vec (WIP)


```python
from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
```


```python
fetch_20newsgroups()['target_names']
```




    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']




```python
data = fetch_20newsgroups(
    categories=['comp.graphics','sci.med','talk.politics.guns','rec.autos','soc.religion.christian']
)
```


```python
news = data['data']
target = data['target']
target_names = data['target_names']
```


```python
model = Top2Vec(news, workers=10)
```

    2021-08-20 11:51:49,718 - top2vec - INFO - Pre-processing documents for training
    2021-08-20 11:51:51,623 - top2vec - INFO - Creating joint document/word embedding
    2021-08-20 11:52:17,552 - top2vec - INFO - Creating lower dimension embedding of documents
    2021-08-20 11:52:26,932 - top2vec - INFO - Finding dense areas of documents
    2021-08-20 11:52:26,992 - top2vec - INFO - Finding topics



```python
model.get_topics()[0][3]
```




    array(['pain', 'symptoms', 'vitamin', 'doctor', 'treatment', 'disease',
           'patients', 'she', 'her', 'diet', 'patient', 'foods', 'infection',
           'physician', 'severe', 'candida', 'skin', 'doctors', 'yeast',
           'cure', 'infections', 'medical', 'zisfein', 'problems', 'seizures',
           'tests', 'weeks', 'months', 'exercise', 'eating', 'anti', 'eat',
           'low', 'prevent', 'hiv', 'friend', 'dsl', 'advice', 'pressure',
           'clinical', 'banks', 'blood', 'syndrome', 'condition', 'gordon',
           'minutes', 'jxp', 'damage', 'cadre', 'food'], dtype='<U14')




```python
model.query_topics('pixel', 5)
```




    ([array(['files', 'image', 'format', 'graphics', 'display', 'pc', 'ftp',
             'formats', 'package', 'interface', 'images', 'dos', 'software',
             'viewer', 'output', 'program', 'mac', 'pub', 'animation',
             'shareware', 'processing', 'gif', 'hardware', 'algorithm', 'disk',
             'jpeg', 'available', 'vga', 'sites', 'bits', 'directory',
             'visualization', 'windows', 'color', 'fax', 'tar', 'interactive',
             'pixel', 'via', 'sgi', 'screen', 'colors', 'data', 'pov',
             'version', 'amiga', 'contact', 'conversion', 'objects', 'user'],
            dtype='<U14'),
      array(['rocks', 'kids', 'warning', 'hit', 'rock', 'violent', 'convex',
             'truck', 'society', 'danger', 'front', 'killed', 'guy', 'stop',
             'low', 'abiding', 'boeing', 'kill', 'sick', 'night', 'decided',
             'drop', 'cars', 'corp', 'protect', 'ago', 'serious', 'car',
             'shoot', 'big', 'gone', 'police', 'utk', 'killing', 'shot', 'user',
             'property', 'accident', 'threat', 'violence', 'innocent', 'five',
             'please', 'defend', 'guns', 'citizens', 'damage', 'criminal',
             'behind', 'vax'], dtype='<U14'),
      array(['cview', 'viewer', 'files', 'disk', 'directory', 'nl', 'windows',
             'dos', 'gif', 'vga', 'jpeg', 'programming', 'version', 'os',
             'shareware', 'convert', 'cd', 'screen', 'display', 'bit', 'stupid',
             'conversion', 'runs', 'file', 'images', 'image', 'zip', 'pl',
             'format', 'vesa', 'view', 'pov', 'color', 'program', 'space',
             'answers', 'sites', 'tin', 'create', 'formats', 'newsreader',
             'pub', 'slow', 'xv', 'programs', 'interface', 'colors',
             'quicktime', 'graphics', 'ftp'], dtype='<U14'),
      array(['tiff', 'ab', 'allen', 'purdue', 'format', 'files', 'unc',
             'writing', 'suggestions', 'image', 'jpeg', 'formats', 'version',
             'images', 'library', 'martin', 'hardware', 'bits', 'output',
             'postscript', 'conversion', 'dos', 'mac', 'gif', 'interface',
             'graphics', 'program', 'convert', 'quoted', 'happy', 'comp',
             'write', 'viewer', 'pixel', 'book', 'algorithm', 'disk',
             'processing', 'ch', 'cc', 'shareware', 'read', 'display', 'xv',
             'pl', 'input', 'zip', 'nearly', 'med', 'copy'], dtype='<U14'),
      array(['urbana', 'cso', 'uiuc', 'uxa', 'illinois', 'darren', 'car',
             'really', 'cars', 'sold', 'top', 'gt', 'boyle', 'looking', 'mary',
             'lights', 'routine', 'engine', 'friend', 'saturn', 'entirely',
             'windows', 'craig', 'nice', 'dividian', 'fine', 'burns', 'bought',
             'dealer', 'il', 'ranch', 'door', 'popular', 'price', 'guess',
             'quite', 'transmission', 'thanks', 'cold', 'east', 'fast', 'honda',
             'keys', 'behind', 'vga', 'survivors', 'virginia', 'polygon',
             'slow', 'looks'], dtype='<U14')],
     [array([0.6021745 , 0.60208344, 0.5625645 , 0.54721   , 0.54593974,
             0.5455748 , 0.5438826 , 0.543128  , 0.54269296, 0.5418372 ,
             0.5337944 , 0.51633537, 0.51296765, 0.50511104, 0.5012145 ,
             0.4996002 , 0.49478245, 0.4934358 , 0.49123   , 0.4875347 ,
             0.4842102 , 0.48241353, 0.47358835, 0.4712616 , 0.47034538,
             0.46958125, 0.4680711 , 0.46246773, 0.46075124, 0.46059793,
             0.45770603, 0.4525558 , 0.44766843, 0.44364196, 0.44352496,
             0.44123134, 0.4399249 , 0.43798912, 0.4294169 , 0.42798984,
             0.42723337, 0.424015  , 0.4225667 , 0.41773134, 0.41711527,
             0.41675615, 0.4129578 , 0.41272712, 0.41027957, 0.40546167],
            dtype=float32),
      array([0.6743512 , 0.60657084, 0.4477735 , 0.4199478 , 0.3907888 ,
             0.37549782, 0.351242  , 0.34446847, 0.33783242, 0.32700366,
             0.32370383, 0.3217173 , 0.32011536, 0.30931515, 0.30624676,
             0.2961492 , 0.2936295 , 0.29295826, 0.29241842, 0.29205543,
             0.2916168 , 0.29039618, 0.28449947, 0.2821859 , 0.28072822,
             0.26894262, 0.26645496, 0.2663145 , 0.2650054 , 0.26235867,
             0.26122707, 0.2586014 , 0.25688016, 0.25330168, 0.24865216,
             0.24611339, 0.24599779, 0.24538921, 0.24366626, 0.24305408,
             0.24299863, 0.2414188 , 0.23854643, 0.23780365, 0.2351208 ,
             0.23292992, 0.23202688, 0.2314744 , 0.2311647 , 0.2307774 ],
            dtype=float32),
      array([0.8070735 , 0.60334194, 0.5474779 , 0.5415514 , 0.49766305,
             0.48404905, 0.43215227, 0.4313448 , 0.41330853, 0.40591764,
             0.40549785, 0.39814585, 0.39520246, 0.38963938, 0.36557025,
             0.35776407, 0.35012418, 0.34909517, 0.34904945, 0.3487044 ,
             0.34350467, 0.34137765, 0.34111023, 0.3398481 , 0.33890432,
             0.33839393, 0.33803558, 0.32730296, 0.32684147, 0.32585087,
             0.32238805, 0.32151455, 0.31705388, 0.31352   , 0.30838156,
             0.2999366 , 0.29844928, 0.29809624, 0.29729795, 0.29229793,
             0.29005167, 0.2893554 , 0.2888195 , 0.28448522, 0.28259462,
             0.28226388, 0.28195974, 0.2777747 , 0.27740425, 0.2760503 ],
            dtype=float32),
      array([0.7090144 , 0.54499936, 0.49833834, 0.48076802, 0.39717746,
             0.38205838, 0.36548868, 0.3585232 , 0.35243052, 0.34398663,
             0.34367812, 0.34128723, 0.32822055, 0.32066357, 0.3078061 ,
             0.30685312, 0.3028316 , 0.29598072, 0.29071355, 0.2893713 ,
             0.28715354, 0.2858879 , 0.28560835, 0.2807576 , 0.27379495,
             0.2706669 , 0.26519564, 0.2643212 , 0.2616133 , 0.26030886,
             0.26021478, 0.25965527, 0.25825572, 0.25595838, 0.25344276,
             0.25156638, 0.25006714, 0.24543604, 0.24314219, 0.2430661 ,
             0.2425078 , 0.24008632, 0.23616987, 0.23613635, 0.23607016,
             0.23593184, 0.2354028 , 0.23530862, 0.23108001, 0.22712599],
            dtype=float32),
      array([0.6890638 , 0.67214614, 0.66952145, 0.62732893, 0.5509174 ,
             0.32844374, 0.29489586, 0.29242846, 0.28789175, 0.28449225,
             0.28403825, 0.27961197, 0.2777422 , 0.27760994, 0.27646035,
             0.27445498, 0.2706978 , 0.25673053, 0.2530842 , 0.24956468,
             0.24088301, 0.23911434, 0.23815548, 0.23590559, 0.23337242,
             0.23126042, 0.22833961, 0.22697851, 0.22650349, 0.22617558,
             0.22453019, 0.22428639, 0.223548  , 0.22206794, 0.21998194,
             0.21991217, 0.21900776, 0.21893473, 0.21586755, 0.21539861,
             0.21459132, 0.21153273, 0.21014714, 0.20917831, 0.20882875,
             0.20824118, 0.20768456, 0.20759207, 0.20503493, 0.203206  ],
            dtype=float32)],
     array([0.39127362, 0.25513235, 0.25318933, 0.23153704, 0.15963803],
           dtype=float32),
     array([ 0, 16, 27, 24, 15]))




```python

```
