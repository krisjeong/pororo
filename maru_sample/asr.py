#!/usr/bin/env python
"""
    ASR sample program
"""
import argparse
import logging

from pororo import Pororo
from maru_utils.logging_utils import init_logger

logger = None


def main(args):
    global logger
    logger = init_logger(args.log_file, log_file_level=logging.DEBUG)
    logger.info(f'args: {args}')

    asr = Pororo(task='asr', lang=args.lang)  # do args.lang to get info from ash file      # args: Namespace(audio_path='korean_sample3.wav', lang='kor', log_file='./logs/asr.log')
    logger.info(f'ASR Result!\n{asr(args.audio_path)}')             # asr(args.audio_path) == asr('korean_sample3.wav') ->


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-audio_path', default='korean_sample3.wav')
    parser.add_argument('-log_file', default='./logs/asr.log')
    parser.add_argument('-lang', default='kor', type=str, choices=['ko', 'en', 'zh'])

    args = parser.parse_args()

    main(args)
