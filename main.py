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
        try:
            method()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f'{method.__name__} done ❌')
            continue
        print(f'{method.__name__} done ✅')


if __name__ == "__main__":
    main()
