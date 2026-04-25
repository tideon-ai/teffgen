"""Integration tests for health checking utilities."""



class TestHealthChecker:
    """Test health checking (no GPU required)."""

    def test_health_check_all(self):
        from teffgen.utils.health import HealthChecker
        checker = HealthChecker()
        results = checker.check_all()
        assert len(results) > 0
        for check in results:
            assert hasattr(check, "name")
            assert hasattr(check, "passed")
            assert hasattr(check, "message")

    def test_website_check(self):
        from teffgen.utils.health import HealthChecker
        checker = HealthChecker()
        result = checker.check_website("https://tideon.ai")
        assert hasattr(result, "passed")
        assert hasattr(result, "message")

    def test_pypi_check(self):
        from teffgen.utils.health import HealthChecker
        checker = HealthChecker()
        result = checker.check_pypi()
        assert hasattr(result, "passed")
        assert hasattr(result, "message")
