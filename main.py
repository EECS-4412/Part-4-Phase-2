from clients.SqliteClient import SqlClient
import traceback
from parts.Part1 import part1
from parts.Part2 import part2
from parts.Part3 import part3

from dotenv import load_dotenv

load_dotenv()


def main():
    methods = [
        part1,
        part2,
        part3,
    ]
    for method in methods:
        print(f'---===== {method.__name__} starting 🤔 =====---\n')
        try:
            method()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f'\n---===== {method.__name__} done ❌=====---\n')
            continue
        print(f'\n---===== {method.__name__} done ✅ =====---\n')


if __name__ == "__main__":
    main()
