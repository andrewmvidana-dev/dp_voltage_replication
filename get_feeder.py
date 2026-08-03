# Downloads the IEEE 123-bus test feeder. Run once, before run_day1.py.
#
# The feeder is four small text files (~39 kB) inside a much larger public
# repository, so we fetch exactly those rather than cloning it. Uses only the
# standard library, so it works before anything else is installed.

import os
import urllib.request
import urllib.error


# The public GitHub folder holding the IEEE test cases. "raw.githubusercontent"
# serves the plain file contents rather than the web page around them.
BASE = (
    "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
    "master/Version8/Distrib/IEEETestCases"
)

# Which files we need, and where each one lives inside that folder.
#
# Note the last entry. Inside the 123Bus folder there IS a file called
# IEEELineCodes.DSS, but it is a 30-byte stub that just points one level up.
# If you download the stub instead of the real thing, OpenDSS fails with
# 'Redirect file not found: "../IEEELineCodes.DSS"'. So we deliberately take
# the real one from the parent folder and save it under the expected name.
FILES = {
    "IEEE123Master.dss":     f"{BASE}/123Bus/IEEE123Master.dss",
    "IEEE123Regulators.DSS": f"{BASE}/123Bus/IEEE123Regulators.DSS",
    "IEEE123Loads.DSS":      f"{BASE}/123Bus/IEEE123Loads.DSS",
    "IEEELineCodes.DSS":     f"{BASE}/IEEELineCodes.DSS",
}

DEST = "feeders"      # folder to create, relative to this script


def main():
    # os.makedirs creates the folder. exist_ok=True means "do nothing if it
    # is already there" rather than raising an error.
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading the IEEE 123-bus feeder into ./{DEST}/\n")

    total = 0
    for filename, url in FILES.items():
        target = os.path.join(DEST, filename)
        try:
            urllib.request.urlretrieve(url, target)
        except urllib.error.URLError as err:
            print(f"  FAILED  {filename}")
            print(f"          {err}")
            print()
            print("  Could not reach GitHub. Check your internet connection.")
            print("  If you are on a university or corporate network, a proxy")
            print("  may be blocking it; try a different connection.")
            return

        size = os.path.getsize(target)
        total += size
        print(f"  ok  {filename:24s} {size:>7,} bytes")

    print(f"\nDone. {len(FILES)} files, {total:,} bytes total.")
    print()
    print("Now open run_day1.py and press Run.")


if __name__ == "__main__":
    main()
