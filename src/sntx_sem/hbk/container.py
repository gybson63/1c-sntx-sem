"""HBK container binary format reader for 1C help files."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

END_MARKER = 0x7FFFFFFF


@dataclass
class BlockHeader:
    payload_size: int
    block_size: int
    next_block: int


class HbkReader:
    """Read entities (PackBlock, FileStorage, Book) from an HBK container."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._entities: dict[str, bytes] = {}
        self._parse()

    @classmethod
    def from_path(cls, path: str | Path) -> HbkReader:
        return cls(Path(path).read_bytes())

    @property
    def entities(self) -> dict[str, bytes]:
        return dict(self._entities)

    def get_entity(self, name: str) -> bytes | None:
        return self._entities.get(name)

    def _parse(self) -> None:
        if len(self._data) < 16:
            raise ValueError("HBK file too small")
        toc_offset = 16
        file_infos = self._read_toc_block(toc_offset)
        for header_addr, body_addr in file_infos:
            name = self._read_file_name(header_addr)
            body = self._read_file_body(body_addr)
            if name:
                self._entities[name] = body

    def _read_toc_block(self, offset: int) -> list[tuple[int, int]]:
        header, data_start = self._read_block_header(offset)
        data = self._read_block_data(data_start, header)
        infos: list[tuple[int, int]] = []
        pos = 0
        while pos + 12 <= len(data):
            header_addr, body_addr, reserved = struct.unpack_from("<3i", data, pos)
            if reserved != END_MARKER:
                break
            infos.append((header_addr, body_addr))
            pos += 12
        return infos

    def _read_file_name(self, header_addr: int) -> str:
        header, data_start = self._read_block_header(header_addr)
        raw = self._read_block_data(data_start, header)
        name_region = raw[20 : header.payload_size]
        if not name_region:
            return ""
        # UTF-16LE null-terminated entity name (PackBlock, FileStorage, …)
        end = 0
        while end + 1 < len(name_region) and name_region[end : end + 2] != b"\x00\x00":
            end += 2
        name_bytes = name_region[:end] if end else name_region
        return name_bytes.decode("utf-16-le", errors="replace").strip("\x00")

    def _read_file_body(self, body_addr: int) -> bytes:
        parts: list[bytes] = []
        offset = body_addr
        while offset != END_MARKER and offset < len(self._data):
            header, data_start = self._read_block_header(offset)
            chunk = self._read_block_data(data_start, header)
            parts.append(chunk)
            if header.next_block == END_MARKER:
                break
            offset = header.next_block
        return b"".join(parts)

    def _read_block_header(self, offset: int) -> tuple[BlockHeader, int]:
        if offset + 31 > len(self._data):
            raise ValueError(f"Block header out of range at {offset}")
        if self._data[offset : offset + 2] != b"\r\n":
            raise ValueError(f"Invalid block CRLF at {offset}")
        payload_size = self._parse_hex_int(self._data[offset + 2 : offset + 10])
        block_size = self._parse_hex_int(self._data[offset + 11 : offset + 19])
        next_block = self._parse_hex_int(self._data[offset + 20 : offset + 28])
        data_start = offset + 31
        return BlockHeader(payload_size, block_size, next_block), data_start

    def _read_block_data(self, data_start: int, header: BlockHeader) -> bytes:
        end = data_start + header.block_size
        if end > len(self._data):
            end = len(self._data)
        return self._data[data_start:end]

    @staticmethod
    def _parse_hex_int(raw: bytes) -> int:
        text = raw.decode("ascii", errors="ignore").strip()
        if not text or text.lower() == "ffffffff":
            return END_MARKER
        return int(text, 16)


def inflate_pack_block(pack_block: bytes) -> bytes:
    """Decompress PackBlock inner zlib payload."""
    with zipfile.ZipFile(io.BytesIO(pack_block)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Empty PackBlock ZIP")
        inner = zf.read(names[0])
    try:
        return zlib.decompress(inner)
    except zlib.error:
        return inner


def open_file_storage(file_storage: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(file_storage))
