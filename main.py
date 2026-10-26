"""Palmimo アプリのサンプル: Portal で設定した値をログに出すだけの最小構成。

ハードウェア（サーボ・カメラ・マイク・スピーカー）には触らないので、どの端末でも動く。
ここを書き換えて自分のアプリにする。
"""

import argparse
import logging
import os
import time
from importlib.metadata import version

logger = logging.getLogger("my_app")


def main() -> None:
    parser = argparse.ArgumentParser(description="Palmimo app template")
    parser.add_argument("--message", default="こんにちは、Palmimo!")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    # Portal のログ画面には標準出力・標準エラーがそのまま出る。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("app id: %s", os.environ.get("PALMIMO_APP_ID", "(Portal の外で実行中)"))
    logger.info("palmimo-sdk: %s", version("palmimo-sdk"))
    for i in range(1, args.repeat + 1):
        logger.info("[%d/%d] %s", i, args.repeat, args.message)
        time.sleep(1)
    logger.info("done")


if __name__ == "__main__":
    main()
