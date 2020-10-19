import argparse
import os.path as osp
import re

try:
    import requests
    from requests.exceptions import SSLError
except ImportError:
    print('Install requests first.')


def get_page(url):
    while True:
        try:
            page = requests.get(url).text
            return page
        except SSLError:
            print(f'Connection error {url}.')
            continue


def get_title(page):
    reg = re.compile(r'<title>(.*?)YouTube', re.S)
    title = reg.findall(page)[0].replace('-', '')
    return title


def parse_args():
    parser = argparse.ArgumentParser(description='Crawler for youtube videos.')
    parser.add_argument('video_ids', type=str, help='source file')
    parser.add_argument('output_path', type=str, help='output file')
    parser.add_argument(
        '--num-worker', type=int, default=8, help='number of workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.video_ids, 'r') as f:
        with open(args.output_path, 'w') as f_out:
            for line in f.readlines():
                video_id, _ = osp.splitext(line.rstrip().split()[0])
                page = get_page(f'https://www.youtube.com/watch?v={video_id}')
                title = get_title(page)
                f_out.write(f'{video_id} {title}\n')
