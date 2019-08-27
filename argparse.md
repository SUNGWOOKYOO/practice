# argparse

argparse 검색


기본적으로 argparse를 import 한 scipt를 실행 `$ python prog.py ` 

할때, -h 같은 option을 제공해준다. `$ python prog.py -h`

각각의 method별 output type은 `whos` 또는 official document 를 확인한다. 

```python
# prog.py
import argparse
parser = argparse.Argumentparser() # Argumentparser object를 만듦
args = parser.parse_args() # namespace 출력 
```



