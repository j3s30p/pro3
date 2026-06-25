from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import discord
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_MARKDOWN_DIR = ROOT_DIR / "data" / "markdown"
DEFAULT_ATTACHMENTS_DIR = ROOT_DIR / "data" / "attachments"


class CrawlError(RuntimeError):
    pass


@dataclass(frozen=True)
class CrawlArgs:
    token: str
    guild_id: int | None
    channel_ids: list[int]
    limit: int | None
    before: datetime | None
    after: datetime | None
    raw_dir: Path
    markdown_dir: Path
    attachments_dir: Path
    write_markdown: bool
    download_attachments: bool
    include_bots: bool
    include_empty: bool


@dataclass(frozen=True)
class ChannelSummary:
    category_id: str | None
    category_name: str | None
    channel_id: str
    channel_name: str
    jsonl_path: str
    markdown_path: str | None
    exported_count: int
    skipped_count: int
    downloaded_attachment_count: int
    failed_attachment_count: int
    status: str
    error: str | None = None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_channel_ids(values: Sequence[str] | None, env_value: str | None) -> list[int]:
    raw_values: list[str] = []
    if env_value:
        raw_values.extend(env_value.split(","))
    for value in values or []:
        raw_values.extend(value.split(","))

    channel_ids: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        stripped = raw_value.strip()
        if not stripped:
            continue
        channel_id = int(stripped)
        if channel_id not in seen:
            seen.add(channel_id)
            channel_ids.append(channel_id)
    return channel_ids


def parse_limit(value: str | None) -> int | None:
    if value is None or value == "":
        return 500

    limit = int(value)
    if limit <= 0:
        return None
    return limit


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().casefold() in {"1", "true", "yes", "y", "on"}


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^\w._-]+", "_", value).strip("._-")
    return safe or "channel"


def category_folder_name(category: Any | None) -> str:
    if category is None:
        return "000_no_category"
    return f"{category.position:03d}_{category.id}_{safe_filename(category.name)}"


def channel_file_stem(channel: Any) -> str:
    return f"{channel.position:03d}_{channel.id}_{safe_filename(channel.name)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Discord guild channel messages into per-channel JSONL files.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("DISCORD_TOKEN"),
        help="Discord bot token. Defaults to DISCORD_TOKEN from .env.",
    )
    parser.add_argument(
        "--guild-id",
        default=os.getenv("DISCORD_GUILD_ID"),
        help="Target guild id. If omitted, the bot must be in exactly one guild.",
    )
    parser.add_argument(
        "--channel-id",
        action="append",
        help=(
            "Target channel id. Repeat this option or pass comma-separated ids. "
            "If omitted, every text channel in the guild is exported."
        ),
    )
    parser.add_argument(
        "--limit",
        default=os.getenv("CRAWL_LIMIT", os.getenv("MAX_HISTORY_MESSAGES", "500")),
        help="Messages per channel. Use 0 to export full available history.",
    )
    parser.add_argument(
        "--after",
        default=os.getenv("CRAWL_AFTER"),
        help="Only export messages after this ISO datetime.",
    )
    parser.add_argument(
        "--before",
        default=os.getenv("CRAWL_BEFORE"),
        help="Only export messages before this ISO datetime.",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("DISCORD_RAW_DIR", str(DEFAULT_RAW_DIR)),
        help="Directory for JSONL exports.",
    )
    parser.add_argument(
        "--markdown-dir",
        default=os.getenv("DISCORD_MARKDOWN_DIR", str(DEFAULT_MARKDOWN_DIR)),
        help="Directory for human-readable markdown exports.",
    )
    parser.add_argument(
        "--attachments-dir",
        default=os.getenv("DISCORD_ATTACHMENTS_DIR", str(DEFAULT_ATTACHMENTS_DIR)),
        help="Directory for downloaded attachment files.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip markdown mirror files.",
    )
    parser.set_defaults(download_attachments=parse_bool(os.getenv("DOWNLOAD_ATTACHMENTS"), True))
    parser.add_argument(
        "--download-attachments",
        dest="download_attachments",
        action="store_true",
        help="Download Discord attachment files. This is the default.",
    )
    parser.add_argument(
        "--no-download-attachments",
        dest="download_attachments",
        action="store_false",
        help="Only store attachment metadata and URLs.",
    )
    parser.add_argument(
        "--include-bots",
        action="store_true",
        help="Include bot-authored messages.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include messages with no text, attachments, or embeds.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CrawlArgs:
    namespace = build_parser().parse_args(argv)
    if not namespace.token:
        raise CrawlError("DISCORD_TOKEN is required. Put it in .env or pass --token.")

    guild_id = int(namespace.guild_id) if namespace.guild_id else None
    channel_ids = parse_channel_ids(namespace.channel_id, os.getenv("DISCORD_CHANNEL_IDS"))

    return CrawlArgs(
        token=namespace.token,
        guild_id=guild_id,
        channel_ids=channel_ids,
        limit=parse_limit(namespace.limit),
        before=parse_datetime(namespace.before),
        after=parse_datetime(namespace.after),
        raw_dir=Path(namespace.raw_dir),
        markdown_dir=Path(namespace.markdown_dir),
        attachments_dir=Path(namespace.attachments_dir),
        write_markdown=not namespace.no_markdown,
        download_attachments=namespace.download_attachments,
        include_bots=namespace.include_bots,
        include_empty=namespace.include_empty,
    )


def attachment_to_record(attachment: discord.Attachment) -> dict[str, Any]:
    return {
        "attachment_id": str(attachment.id),
        "filename": attachment.filename,
        "description": getattr(attachment, "description", None),
        "url": attachment.url,
        "proxy_url": attachment.proxy_url,
        "content_type": attachment.content_type,
        "size": attachment.size,
        "height": getattr(attachment, "height", None),
        "width": getattr(attachment, "width", None),
        "duration": getattr(attachment, "duration", None),
        "ephemeral": getattr(attachment, "ephemeral", False),
        "spoiler": attachment.is_spoiler(),
        "local_path": None,
        "download_status": "not_requested",
        "download_error": None,
    }


def embed_to_record(embed: discord.Embed) -> dict[str, Any]:
    raw = embed.to_dict()
    text_parts: list[str] = []

    for key in ("title", "description", "url"):
        value = raw.get(key)
        if value:
            text_parts.append(str(value))

    author = raw.get("author") or {}
    if author.get("name"):
        text_parts.append(str(author["name"]))

    provider = raw.get("provider") or {}
    if provider.get("name"):
        text_parts.append(str(provider["name"]))

    footer = raw.get("footer") or {}
    if footer.get("text"):
        text_parts.append(str(footer["text"]))

    for field in raw.get("fields", []):
        if field.get("name"):
            text_parts.append(str(field["name"]))
        if field.get("value"):
            text_parts.append(str(field["value"]))

    for key in ("image", "thumbnail", "video"):
        media = raw.get(key) or {}
        if media.get("url"):
            text_parts.append(str(media["url"]))

    return {
        "type": raw.get("type", embed.type),
        "title": raw.get("title"),
        "description": raw.get("description"),
        "url": raw.get("url"),
        "text": "\n".join(text_parts),
        "raw": raw,
    }


def message_to_record(message: discord.Message) -> dict[str, Any]:
    guild = message.guild
    channel = message.channel
    category = getattr(channel, "category", None)
    reference = message.reference

    return {
        "guild_id": str(guild.id) if guild else None,
        "guild_name": guild.name if guild else None,
        "channel_id": str(channel.id),
        "channel_name": getattr(channel, "name", "unknown"),
        "category_id": str(category.id) if category else None,
        "category_name": category.name if category else None,
        "message_id": str(message.id),
        "message_type": str(message.type),
        "author_id": str(message.author.id),
        "author_name": message.author.name,
        "author_display_name": getattr(message.author, "display_name", message.author.name),
        "author_global_name": getattr(message.author, "global_name", None),
        "author_is_bot": message.author.bot,
        "content": message.content,
        "clean_content": message.clean_content,
        "created_at": message.created_at.isoformat(timespec="seconds"),
        "edited_at": message.edited_at.isoformat(timespec="seconds") if message.edited_at else None,
        "jump_url": message.jump_url,
        "referenced_message_id": str(reference.message_id) if reference and reference.message_id else None,
        "mentions": [
            {
                "user_id": str(user.id),
                "name": user.name,
                "display_name": getattr(user, "display_name", user.name),
            }
            for user in message.mentions
        ],
        "role_mentions": [
            {
                "role_id": str(role.id),
                "name": role.name,
            }
            for role in message.role_mentions
        ],
        "attachments": [attachment_to_record(attachment) for attachment in message.attachments],
        "embeds": [embed_to_record(embed) for embed in message.embeds],
    }


def should_export_record(record: dict[str, Any], include_empty: bool) -> bool:
    if include_empty:
        return True
    return bool(record["content"].strip() or record["attachments"] or record["embeds"])


def format_markdown_message(record: dict[str, Any]) -> str:
    author = record["author_display_name"] or record["author_name"]
    created_at = record["created_at"]
    content = record["clean_content"] or record["content"] or "(내용 없음)"

    lines = [
        f"## {created_at} | {author}",
        "",
        content,
        "",
        f"- message_id: {record['message_id']}",
        f"- author_id: {record['author_id']}",
        f"- jump_url: {record['jump_url']}",
    ]

    if record["referenced_message_id"]:
        lines.append(f"- referenced_message_id: {record['referenced_message_id']}")

    if record["attachments"]:
        lines.append("- attachments:")
        for attachment in record["attachments"]:
            local_path = attachment.get("local_path") or "not_downloaded"
            lines.append(f"  - {attachment['filename']}: {attachment['url']}")
            lines.append(f"    - local_path: {local_path}")
            lines.append(f"    - download_status: {attachment.get('download_status')}")

    if record["embeds"]:
        lines.append("- embeds:")
        for index, embed in enumerate(record["embeds"], start=1):
            lines.append(f"  - embed_{index}:")
            if embed.get("title"):
                lines.append(f"    - title: {embed['title']}")
            if embed.get("description"):
                lines.append(f"    - description: {embed['description']}")
            if embed.get("url"):
                lines.append(f"    - url: {embed['url']}")

    return "\n".join(lines) + "\n"


async def download_message_attachments(
    message: discord.Message,
    record: dict[str, Any],
    message_attachments_dir: Path,
    enabled: bool,
) -> tuple[int, int]:
    if not enabled:
        return 0, 0

    downloaded_count = 0
    failed_count = 0
    for index, attachment in enumerate(message.attachments):
        attachment_record = record["attachments"][index]
        target_path = message_attachments_dir / (
            f"{message.id}_{attachment.id}_{safe_filename(attachment.filename)}"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            await attachment.save(target_path)
        except discord.HTTPException as exc:
            failed_count += 1
            attachment_record["download_status"] = "failed"
            attachment_record["download_error"] = str(exc)
        else:
            downloaded_count += 1
            attachment_record["local_path"] = str(target_path)
            attachment_record["download_status"] = "downloaded"
            attachment_record["download_error"] = None

    return downloaded_count, failed_count


def resolve_guild(client: discord.Client, guild_id: int | None) -> discord.Guild:
    if guild_id is not None:
        guild = client.get_guild(guild_id)
        if guild is None:
            joined = ", ".join(f"{item.name}({item.id})" for item in client.guilds)
            raise CrawlError(f"Guild {guild_id} was not found. Joined guilds: {joined}")
        return guild

    if len(client.guilds) == 1:
        return client.guilds[0]

    joined = ", ".join(f"{item.name}({item.id})" for item in client.guilds)
    raise CrawlError(f"--guild-id is required when the bot is in multiple guilds. Joined guilds: {joined}")


def resolve_channels(guild: discord.Guild, channel_ids: Sequence[int]) -> list[discord.TextChannel]:
    if not channel_ids:
        return sorted(
            guild.text_channels,
            key=lambda channel: (
                channel.category.position if channel.category else -1,
                channel.position,
                channel.id,
            ),
        )

    channels: list[discord.TextChannel] = []
    for channel_id in channel_ids:
        channel = guild.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            raise CrawlError(f"Channel {channel_id} was not found as a text channel in {guild.name}.")
        channels.append(channel)
    return channels


async def crawl_channel(
    channel: discord.TextChannel,
    args: CrawlArgs,
    guild_raw_dir: Path,
    guild_markdown_dir: Path,
) -> ChannelSummary:
    category = channel.category
    category_dir_name = category_folder_name(category)
    channel_stem = channel_file_stem(channel)
    raw_category_dir = guild_raw_dir / category_dir_name
    markdown_category_dir = guild_markdown_dir / category_dir_name
    attachments_channel_dir = args.attachments_dir / str(channel.guild.id) / category_dir_name / channel_stem
    raw_category_dir.mkdir(parents=True, exist_ok=True)
    if args.write_markdown:
        markdown_category_dir.mkdir(parents=True, exist_ok=True)

    base_name = channel_stem
    jsonl_path = raw_category_dir / f"{base_name}.jsonl"
    markdown_path = markdown_category_dir / f"{base_name}.md" if args.write_markdown else None

    exported_count = 0
    skipped_count = 0
    downloaded_attachment_count = 0
    failed_attachment_count = 0

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            markdown_file = markdown_path.open("w", encoding="utf-8") if markdown_path else None
            try:
                if markdown_file:
                    markdown_file.write(f"# #{channel.name}\n\n")
                    markdown_file.write(f"- category_id: {category.id if category else ''}\n")
                    markdown_file.write(f"- category_name: {category.name if category else 'no_category'}\n")
                    markdown_file.write(f"- channel_id: {channel.id}\n")
                    markdown_file.write(f"- exported_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")

                async for message in channel.history(
                    limit=args.limit,
                    before=args.before,
                    after=args.after,
                    oldest_first=True,
                ):
                    if message.author.bot and not args.include_bots:
                        skipped_count += 1
                        continue

                    record = message_to_record(message)
                    if not should_export_record(record, args.include_empty):
                        skipped_count += 1
                        continue

                    downloaded, failed = await download_message_attachments(
                        message,
                        record,
                        attachments_channel_dir,
                        args.download_attachments,
                    )
                    downloaded_attachment_count += downloaded
                    failed_attachment_count += failed

                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if markdown_file:
                        markdown_file.write(format_markdown_message(record))
                        markdown_file.write("\n---\n\n")
                    exported_count += 1
            finally:
                if markdown_file:
                    markdown_file.close()
    except discord.Forbidden as exc:
        return ChannelSummary(
            category_id=str(category.id) if category else None,
            category_name=category.name if category else None,
            channel_id=str(channel.id),
            channel_name=channel.name,
            jsonl_path=str(jsonl_path),
            markdown_path=str(markdown_path) if markdown_path else None,
            exported_count=exported_count,
            skipped_count=skipped_count,
            downloaded_attachment_count=downloaded_attachment_count,
            failed_attachment_count=failed_attachment_count,
            status="forbidden",
            error=str(exc),
        )
    except discord.HTTPException as exc:
        return ChannelSummary(
            category_id=str(category.id) if category else None,
            category_name=category.name if category else None,
            channel_id=str(channel.id),
            channel_name=channel.name,
            jsonl_path=str(jsonl_path),
            markdown_path=str(markdown_path) if markdown_path else None,
            exported_count=exported_count,
            skipped_count=skipped_count,
            downloaded_attachment_count=downloaded_attachment_count,
            failed_attachment_count=failed_attachment_count,
            status="http_error",
            error=str(exc),
        )

    return ChannelSummary(
        category_id=str(category.id) if category else None,
        category_name=category.name if category else None,
        channel_id=str(channel.id),
        channel_name=channel.name,
        jsonl_path=str(jsonl_path),
        markdown_path=str(markdown_path) if markdown_path else None,
        exported_count=exported_count,
        skipped_count=skipped_count,
        downloaded_attachment_count=downloaded_attachment_count,
        failed_attachment_count=failed_attachment_count,
        status="ok",
    )


def write_manifest(
    guild: discord.Guild,
    args: CrawlArgs,
    guild_raw_dir: Path,
    summaries: list[ChannelSummary],
    started_at: datetime,
) -> Path:
    manifest_path = guild_raw_dir / "manifest.json"
    payload = {
        "guild_id": str(guild.id),
        "guild_name": guild.name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limit": args.limit,
        "after": args.after.isoformat(timespec="seconds") if args.after else None,
        "before": args.before.isoformat(timespec="seconds") if args.before else None,
        "include_bots": args.include_bots,
        "include_empty": args.include_empty,
        "download_attachments": args.download_attachments,
        "attachments_dir": str(args.attachments_dir),
        "channels": [
            {
                "category_id": summary.category_id,
                "category_name": summary.category_name,
                "channel_id": summary.channel_id,
                "channel_name": summary.channel_name,
                "jsonl_path": summary.jsonl_path,
                "markdown_path": summary.markdown_path,
                "exported_count": summary.exported_count,
                "skipped_count": summary.skipped_count,
                "downloaded_attachment_count": summary.downloaded_attachment_count,
                "failed_attachment_count": summary.failed_attachment_count,
                "status": summary.status,
                "error": summary.error,
            }
            for summary in summaries
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


async def run_crawl(args: CrawlArgs) -> None:
    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        started_at = datetime.now(timezone.utc)
        try:
            guild = resolve_guild(client, args.guild_id)
            channels = resolve_channels(guild, args.channel_ids)
            guild_raw_dir = args.raw_dir / str(guild.id)
            guild_markdown_dir = args.markdown_dir / str(guild.id)
            guild_raw_dir.mkdir(parents=True, exist_ok=True)
            if args.write_markdown:
                guild_markdown_dir.mkdir(parents=True, exist_ok=True)

            print(f"Connected as {client.user}. Exporting {len(channels)} channel(s) from {guild.name}.")

            summaries: list[ChannelSummary] = []
            for channel in channels:
                print(f"- Exporting #{channel.name} ({channel.id})")
                summary = await crawl_channel(channel, args, guild_raw_dir, guild_markdown_dir)
                summaries.append(summary)
                status = f"{summary.status}, exported={summary.exported_count}, skipped={summary.skipped_count}"
                print(f"  {status}")

            manifest_path = write_manifest(guild, args, guild_raw_dir, summaries, started_at)
            print(f"Manifest written: {manifest_path}")
        except Exception as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            raise
        finally:
            await client.close()

    await client.start(args.token)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(ROOT_DIR / ".env")
    try:
        args = parse_args(argv)
        asyncio.run(run_crawl(args))
    except CrawlError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
