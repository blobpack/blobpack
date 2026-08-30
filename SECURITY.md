# Security

## Reporting

Report suspected vulnerabilities through GitHub's private advisory form
under this repository's Security tab. Please do not open a public issue
first.

## Scope

Blob Pack reads archives that a dataset author supplies, so a malicious
pack is a real input. Readers reject members that are compressed,
encrypted, or whose declared byte range falls outside the archive, and
member names are refused if they escape the extraction root. A pack that
gets past those checks and causes a read outside its own bounds, a write
outside the destination directory, or arbitrary code execution is a
vulnerability — please report it.

Direct-offset reads skip per-read CRC verification by design: that is the
speed the format exists for. Corruption is caught by `blobpack verify`,
not by every read, so a checksum mismatch on damaged data is expected
behaviour rather than a vulnerability.
