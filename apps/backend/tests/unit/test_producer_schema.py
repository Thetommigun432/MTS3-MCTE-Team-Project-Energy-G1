"""
Unit tests for data producer payload schema.

Tests that the producer generates data in the correct format
for downstream Redis consumers and InfluxDB writers.
"""

import json
import pytest



class TestSimulationPayloadSchema:
    """Tests for simulated data payload format."""

    @pytest.mark.unit
    def test_simulation_payload_has_required_keys(self):
        """Simulated data generates payload with all required keys."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        required_keys = {"timestamp", "power_total", "voltage", "current", "power_factor"}
        assert set(payload.keys()) == required_keys

    @pytest.mark.unit
    def test_simulation_payload_types(self):
        """Simulated data has correct types for all fields."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        assert isinstance(payload["timestamp"], float)
        assert isinstance(payload["power_total"], float)
        assert isinstance(payload["voltage"], float)
        assert isinstance(payload["current"], float)
        assert isinstance(payload["power_factor"], float)

    @pytest.mark.unit
    def test_simulation_power_positive(self):
        """Simulated power values are always positive."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()

        # Check first 100 samples
        for _ in range(100):
            payload = next(gen)
            assert payload["power_total"] > 0, "Power must be positive"

    @pytest.mark.unit
    def test_simulation_voltage_realistic(self):
        """Simulated voltage is in realistic EU range."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()

        for _ in range(50):
            payload = next(gen)
            # EU voltage is 230V +/- 10% = 207-253V, with some noise
            assert 200 < payload["voltage"] < 260, f"Voltage {payload['voltage']} out of range"

    @pytest.mark.unit
    def test_simulation_current_consistent_with_power(self):
        """Current should be approximately power/voltage."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        expected_current = payload["power_total"] / payload["voltage"]
        # Allow some rounding tolerance
        assert abs(payload["current"] - expected_current) < 0.1

    @pytest.mark.unit
    def test_simulation_power_factor_reasonable(self):
        """Power factor should be between 0.8 and 1.0."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()

        for _ in range(50):
            payload = next(gen)
            assert 0.8 < payload["power_factor"] <= 1.0


class TestPayloadJsonSerialization:
    """Tests for JSON serialization of payloads."""

    @pytest.mark.unit
    def test_simulation_payload_json_serializable(self):
        """All simulation payload values are JSON serializable."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        # Should not raise
        serialized = json.dumps(payload)
        deserialized = json.loads(serialized)

        # Values should survive round-trip
        assert deserialized["timestamp"] == payload["timestamp"]
        assert deserialized["power_total"] == payload["power_total"]

    @pytest.mark.unit
    def test_payload_no_nan_values(self):
        """Payloads should never contain NaN values."""
        from app.pipeline.producer import simulate_building_power
        import math

        gen = simulate_building_power()

        for _ in range(100):
            payload = next(gen)
            for key, value in payload.items():
                assert not math.isnan(value), f"{key} contains NaN"

    @pytest.mark.unit
    def test_payload_no_inf_values(self):
        """Payloads should never contain infinite values."""
        from app.pipeline.producer import simulate_building_power
        import math

        gen = simulate_building_power()

        for _ in range(100):
            payload = next(gen)
            for key, value in payload.items():
                assert not math.isinf(value), f"{key} contains infinity"


class TestParquetPayloadSchema:
    """Tests for parquet-based data payload format."""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a minimal parquet file for testing."""
        import pandas as pd

        df = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "Aggregate": [1.5, 2.0, 1.8, 2.1, 1.9, 2.2, 1.7, 2.3, 2.0, 1.6],  # kW
        })

        parquet_path = tmp_path / "test_data.parquet"
        df.to_parquet(parquet_path)
        return parquet_path

    @pytest.mark.unit
    def test_parquet_payload_has_required_keys(self, sample_parquet):
        """Parquet reader generates payload with all required keys."""
        from app.pipeline.producer import read_parquet_file

        gen = read_parquet_file(str(sample_parquet))
        payload = next(gen)

        required_keys = {"timestamp", "power_total", "voltage", "current"}
        assert required_keys.issubset(set(payload.keys()))

    @pytest.mark.unit
    def test_parquet_kw_to_watts_conversion(self, sample_parquet):
        """Parquet reader converts kW to Watts correctly."""
        from app.pipeline.producer import read_parquet_file

        gen = read_parquet_file(str(sample_parquet))
        payload = next(gen)

        # First row has 1.5 kW = 1500 W
        assert payload["power_total"] == 1500.0

    @pytest.mark.unit
    def test_parquet_timestamp_extraction(self, sample_parquet):
        """Parquet reader extracts timestamp correctly."""
        from app.pipeline.producer import read_parquet_file

        gen = read_parquet_file(str(sample_parquet))
        payload = next(gen)

        # Timestamp should be a float (Unix timestamp)
        assert isinstance(payload["timestamp"], float)
        assert payload["timestamp"] > 0

    @pytest.mark.unit
    def test_parquet_voltage_default(self, sample_parquet):
        """Parquet reader uses default voltage of 230V."""
        from app.pipeline.producer import read_parquet_file

        gen = read_parquet_file(str(sample_parquet))
        payload = next(gen)

        assert payload["voltage"] == 230.0

    @pytest.mark.unit
    def test_parquet_current_calculated(self, sample_parquet):
        """Parquet reader calculates current from power and voltage."""
        from app.pipeline.producer import read_parquet_file

        gen = read_parquet_file(str(sample_parquet))
        payload = next(gen)

        expected_current = payload["power_total"] / 230.0
        assert abs(payload["current"] - expected_current) < 0.01


class TestCSVPayloadSchema:
    """Tests for CSV-based data payload format."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a minimal CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text(
            "timestamp,power_total,voltage,current\n"
            "1704067200.0,1500.0,230.5,6.5\n"
            "1704067260.0,1600.0,229.8,6.96\n"
        )
        return csv_path

    @pytest.mark.unit
    def test_csv_payload_has_required_keys(self, sample_csv):
        """CSV reader generates payload with all required keys."""
        from app.pipeline.producer import read_csv_file

        gen = read_csv_file(str(sample_csv))
        payload = next(gen)

        required_keys = {"timestamp", "power_total", "voltage", "current"}
        assert required_keys.issubset(set(payload.keys()))

    @pytest.mark.unit
    def test_csv_payload_types(self, sample_csv):
        """CSV reader converts string values to proper types."""
        from app.pipeline.producer import read_csv_file

        gen = read_csv_file(str(sample_csv))
        payload = next(gen)

        assert isinstance(payload["timestamp"], float)
        assert isinstance(payload["power_total"], float)
        assert isinstance(payload["voltage"], float)
        assert isinstance(payload["current"], float)

    @pytest.mark.unit
    def test_csv_values_correct(self, sample_csv):
        """CSV reader parses values correctly."""
        from app.pipeline.producer import read_csv_file

        gen = read_csv_file(str(sample_csv))
        payload = next(gen)

        assert payload["timestamp"] == 1704067200.0
        assert payload["power_total"] == 1500.0
        assert payload["voltage"] == 230.5


class TestRedisMessageFormat:
    """Tests for Redis message format compatibility."""

    @pytest.mark.unit
    def test_payload_suitable_for_redis_publish(self):
        """Payload can be serialized for Redis PUBLISH command."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        # Redis PUBLISH expects bytes or string
        message = json.dumps(payload)
        assert isinstance(message, str)

        # Should be deserializable
        recovered = json.loads(message)
        assert recovered == payload

    @pytest.mark.unit
    def test_payload_compatible_with_stream_xadd(self):
        """Payload structure is compatible with Redis Stream XADD."""
        from app.pipeline.producer import simulate_building_power

        gen = simulate_building_power()
        payload = next(gen)

        # Redis Streams require string keys and values
        # When using XADD with JSON, we'd do XADD stream * data {json}
        # All our values should be convertible to strings
        for key, value in payload.items():
            str_key = str(key)
            str_value = str(value)
            assert len(str_key) > 0
            assert len(str_value) > 0
