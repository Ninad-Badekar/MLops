from src.aws_artifacts import parse_blob_uri


def test_parse_blob_uri():
    account, container, blob = parse_blob_uri(
        "https://myaccount.blob.core.windows.net/artifacts/serving/model.pkl"
    )
    assert account == "myaccount"
    assert container == "artifacts"
    assert blob == "serving/model.pkl"


def test_parse_blob_uri_invalid_scheme():
    try:
        parse_blob_uri("s3://some-bucket/key")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_parse_blob_uri_missing_blob():
    try:
        parse_blob_uri("https://myaccount.blob.core.windows.net/artifacts/")
        assert False, "expected ValueError"
    except ValueError:
        pass
