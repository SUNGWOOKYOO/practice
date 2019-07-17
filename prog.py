import argparse

# basic
# python prog.py -h 기능 제공 
parser = argparse.ArgumentParser() # ArgumentParser 객체 생성
# args = parser.parse_args() # namespace 출력

# add_argument("<arg_이름>", <option 변수>=" option을 주었을떄 출력하고픈 문자열")
# ArgumentParser 객체에 arg_이름 변수가 생기고 값을 받을 수 있다. 
parser.add_argument("--verbosity", help="increase output verbosity")
args = parser.parse_args()

# if `$ python prog.py --verbosity 23`, then args.verbosity = 23
if args.verbosity:
	print("verbosity value: ", args.verbosity) 
