from __future__ import annotations

from types import SimpleNamespace

from discord_crawler.crawl import (
    category_folder_name,
    channel_file_stem,
    parse_bool,
    parse_channel_ids,
    parse_limit,
    safe_filename,
)


def test_parse_channel_ids_from_env_and_cli_without_duplicates() -> None:
    channel_ids = parse_channel_ids(["3, 4", "5"], "1,2,3")

    assert channel_ids == [1, 2, 3, 4, 5]


def test_parse_limit_zero_means_full_history() -> None:
    assert parse_limit("0") is None


def test_parse_bool_accepts_common_truthy_values() -> None:
    assert parse_bool("true") is True
    assert parse_bool("ON") is True
    assert parse_bool("", default=True) is True
    assert parse_bool("false", default=True) is False


def test_safe_filename_removes_discord_channel_punctuation() -> None:
    assert safe_filename("공지/회의 room") == "공지_회의_room"


def test_category_folder_name_uses_position_id_and_name() -> None:
    category = SimpleNamespace(position=7, id=123, name="프로젝트/회의")

    assert category_folder_name(category) == "007_123_프로젝트_회의"


def test_category_folder_name_for_uncategorized_channels() -> None:
    assert category_folder_name(None) == "000_no_category"


def test_channel_file_stem_uses_position_id_and_name() -> None:
    channel = SimpleNamespace(position=12, id=456, name="잡담/질문")

    assert channel_file_stem(channel) == "012_456_잡담_질문"
