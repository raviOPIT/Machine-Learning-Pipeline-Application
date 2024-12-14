def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            print(f"✔️ Test Passed: {report.nodeid}")
        elif report.failed:
            print(f"❌ Test Failed: {report.nodeid}")
        elif report.skipped:
            print(f"⚠️ Test Skipped: {report.nodeid}")