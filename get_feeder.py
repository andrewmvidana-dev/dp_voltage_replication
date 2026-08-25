# Downloads the IEEE 123-bus test feeder into ./feeders/. Run once, first.
#
# The feeder is four small text files (~39 kB) inside a much larger public
# repository, so we fetch exactly those rather than cloning it. Standard library
# only, so it works before anything else is installed.

import os
import urllib.request
import urllib.error


BASE = (
    "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
    "master/Version8/Distrib/IEEETestCases"
)

# Note the last entry. The 123Bus folder does contain an IEEELineCodes.DSS, but
# it is a 30-byte stub pointing one level up; download it and OpenDSS fails with
# 'Redirect file not found: "../IEEELineCodes.DSS"'. So we take the real file
# from the parent folder and save it under the expected name.
FILES = {
    "IEEE123Master.dss":     f"{BASE}/123Bus/IEEE123Master.dss",
    "IEEE123Regulators.DSS": f"{BASE}/123Bus/IEEE123Regulators.DSS",
    "IEEE123Loads.DSS":      f"{BASE}/123Bus/IEEE123Loads.DSS",
    "IEEELineCodes.DSS":     f"{BASE}/IEEELineCodes.DSS",
}

DEST = "feeders"


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading the IEEE 123-bus feeder into ./{DEST}/\n")

    total = 0
    for filename, url in FILES.items():
        target = os.path.join(DEST, filename)
        try:
            urllib.request.urlretrieve(url, target)
        except urllib.error.URLError as err:
            print(f"  FAILED  {filename}\n          {err}\n")
            print("  Could not reach GitHub. On a university or corporate")
            print("  network a proxy may be blocking it; try another connection.")
            return

        size = os.path.getsize(target)
        total += size
        print(f"  ok  {filename:24s} {size:>7,} bytes")

    print(f"\nDone. {len(FILES)} files, {total:,} bytes total.")
    print("Next: run_day1.py")


if __name__ == "__main__":
    main()
