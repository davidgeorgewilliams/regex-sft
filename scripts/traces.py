"""Authored reasoning traces -- several phrasings per base turn.

Each base turn has 2-3 alternative phrasings. The build picks one by variant
index, which cuts how often the model sees identical target text: with a single
phrasing per turn, each trace was reused ~7.7x across variants of its concept,
which is real memorisation pressure even though the prompts differ.

The three phrasings are deliberately different in approach rather than being
reworded copies -- typically one reasons forward from the stated rule, one
leads with the construction being used, and one leads with the mistake the
constraint exists to prevent.

Traces are TEMPLATES: {p} is filled with the gold pattern and {r} with the gold
replacement at build time, so a trace cannot commit to a pattern that differs
from the verified answer.

Traces reason from the INSTRUCTION and never cite test strings. A trace quoting
a held string would teach the model to reference evidence it cannot see at
inference time.
"""

CONCEPT_TRACES = {
    # ---------------------------------------------------------- validate
    "us_zip": [
        "The rule pins the length at exactly five digits. Under fullmatch the whole string must be "
        "consumed, so one counted quantifier does it and anything shorter or longer fails. Pattern: {p}",
        "A counted quantifier is the whole construction here: five digits, no anchors needed because "
        "fullmatch already requires the entire string to be covered. Pattern: {p}",
        "If I used '+' instead of a count, any length of digit run would pass. The rule fixes the "
        "length, so the quantifier has to be exact. Pattern: {p}",
    ],
    "us_zip_plus4": [
        "Two shapes are legal: five digits, or five digits plus a hyphen and four more. That is a "
        "required part and an optional suffix in a non-capturing group. Pattern: {p}",
        "I write the base as a counted run, then append the extension as an optional group. Both "
        "lengths are exact, so counted quantifiers rather than '+'. Pattern: {p}",
        "Making the extension mandatory would reject the plain five-digit form, and using '+' inside "
        "it would accept the wrong number of digits. Optional group, exact counts. Pattern: {p}",
    ],
    "email_simple": [
        "A structural check, not a real parser: runs of characters that are neither whitespace nor "
        "'@', joined by an '@', with a literal dot forcing the domain to contain one. Pattern: {p}",
        "Negated classes do the work here. Excluding whitespace and '@' from each run enforces both "
        "conditions on the parts at once, and the escaped dot is required. Pattern: {p}",
        "Using a dot-star for the parts would let whitespace and extra '@' characters through. A "
        "negated class per run prevents that and gives the domain its required dot. Pattern: {p}",
    ],
    "hex_color": [
        "Only two lengths are valid, three or six, so an alternation rather than a range, which would "
        "wrongly admit four and five. The alphabet is digits plus a-f in either case. Pattern: {p}",
        "I branch on the two legal lengths and share the hex class between them. Pattern: {p}",
        "A range quantifier from three to six would accept four and five digits, which are not colours. "
        "Two explicit counted branches avoid that. Pattern: {p}",
    ],
    "hex_literal": [
        "The prefix is mandatory in either case, followed by one or more hex digits. Pattern: {p}",
        "Both prefix spellings collapse into a character class, then '+' over the hex alphabet. "
        "Pattern: {p}",
        "Using '*' after the prefix would accept a bare prefix with no digits at all, so the "
        "quantifier has to be '+'. Pattern: {p}",
    ],
    "iso_date_loose": [
        "The instruction explicitly says not to check calendar values, only the shape. So this is "
        "purely fixed-width fields separated by hyphens. Pattern: {p}",
        "Three counted digit runs joined by literal hyphens. The counts also rule out unpadded "
        "components without a separate rule. Pattern: {p}",
        "Adding month and day range checks would over-constrain this: the task asks for shape only, "
        "and the fixed widths already reject unpadded values. Pattern: {p}",
    ],
    "iso_date_strict": [
        "Unlike the shape-only variant, the ranges matter. Month 01-12 splits into '0' with 1-9 or "
        "'1' with 0-2; day 01-31 into 0[1-9], then [12] with any digit, then 30 and 31. Pattern: {p}",
        "Each range becomes an alternation of two-character branches, which enforces the zero padding "
        "as a side effect of every branch being the same width. Pattern: {p}",
        "Two loose digits per field would accept an impossible month or day. Enumerating the valid "
        "range branches is what prevents that. Pattern: {p}",
    ],
    "leap_day": [
        "The century rule is what bites. Divisible by 4 is a leap year unless divisible by 100, in "
        "which case it must also be divisible by 400. Pattern: {p}",
        "In digits that is two leading digits followed by a two-digit multiple of 4, or a "
        "multiple-of-4 century followed by two zeros. Pattern: {p}",
        "The naive divisible-by-4 pattern accepts century years that are not leap years. Splitting "
        "the century case out is the fix. Pattern: {p}",
    ],
    "semver": [
        "Three numeric components plus an optional pre-release. The leading-zero rejection shapes it: "
        "each component is either exactly '0' or a nonzero digit followed by anything. Pattern: {p}",
        "I build each component as an alternation so a bare zero stays legal while a padded one does "
        "not, then attach the optional hyphenated suffix. Pattern: {p}",
        "A plain digit run per component would accept a padded value, since a quantifier does not "
        "care about a leading zero. That is the trap this pattern is built around. Pattern: {p}",
    ],
    "ipv4_strict": [
        "Each octet is 0-255 with no leading zeros, so I enumerate the ranges downward from 250-255 to "
        "single digits. The dot must be escaped. Pattern: {p}",
        "One octet definition, then three repetitions of dot-plus-octet, which gets the count right by "
        "construction rather than by writing it four times. Pattern: {p}",
        "Three loose digits per octet would accept values above 255, and an unescaped dot would match "
        "any character. Both are avoided here. Pattern: {p}",
    ],
    "ipv4_lenient": [
        "Same 0-255 bound as the strict form, but padding is explicitly allowed, so the low branch "
        "permits an optional leading 0 or 1 before one or two more digits. Pattern: {p}",
        "The only change from the strict version is the low branch widening to admit padded octets "
        "while the upper bound stays at 255. Pattern: {p}",
        "Rejecting leading zeros here would be wrong: the instruction states padded octets are valid, "
        "so the strict construction has to be relaxed. Pattern: {p}",
    ],
    "mac_address": [
        "Six pairs of hex digits joined by colons. Pattern: {p}",
        "I repeat 'pair plus colon' five times and append a bare pair, which fixes the separator count "
        "by construction. Pattern: {p}",
        "Writing the pair six times with colons between is error-prone on the separator count; "
        "factoring it as five repeats plus a tail is safer. Pattern: {p}",
    ],
    "uuid_v4": [
        "Segment lengths are 8-4-4-4-12. Version 4 pins the third group's first character to '4', and "
        "the variant pins the fourth group's to 8, 9, a or b. Pattern: {p}",
        "Counted hex runs give the segment lengths; two pinned nibbles distinguish this from a generic "
        "UUID shape. Pattern: {p}",
        "Without the version and variant nibbles this would accept any UUID, which is exactly what the "
        "instruction rules out. Pattern: {p}",
    ],
    "strong_password": [
        "Four conditions must hold at once, which is what lookaheads are for: each scans from the "
        "start without consuming. Then a restricted alphabet with a minimum length. Pattern: {p}",
        "Assertions first, consumption second. The lookaheads test presence; the trailing class does "
        "the actual matching and bounds the length. Pattern: {p}",
        "Trying to express 'contains one of each' by consuming characters in order would fail, because "
        "the required characters can appear in any order. Lookaheads solve that. Pattern: {p}",
    ],
    "time24": [
        "Hours 00-23 split into [01] with any digit and '2' with 0-3; minutes are 0-5 with any digit. "
        "Pattern: {p}",
        "Both halves are exactly two characters wide, so the zero padding falls out of the range "
        "branches rather than needing its own rule. Pattern: {p}",
        "Two loose digits for the hour would accept values past 23, and allowing a single digit would "
        "break the padding requirement. Pattern: {p}",
    ],
    "time12": [
        "The hour range is 1-12 with an optional leading zero, so either an optional '0' with 1-9, or "
        "'1' with 0-2. The AM/PM suffix is a literal alternation. Pattern: {p}",
        "Optional padding on the hour, standard minute range, then a required space and an uppercase "
        "meridiem alternation. Pattern: {p}",
        "Reusing the 24-hour construction would accept hours above 12, and forgetting the suffix would "
        "accept a bare time. Pattern: {p}",
    ],
    "identifier": [
        "The first character obeys a different rule from the rest: letter or underscore only, no digit. "
        "Pattern: {p}",
        "One class for the head, a wider class for the tail with '*' so single-character names still "
        "work. Pattern: {p}",
        "Using one class for the whole string would accept a leading digit, which the rule forbids. "
        "The head has to be separated out. Pattern: {p}",
    ],
    "username_bounded": [
        "Three constraints: no leading digit, an alphabet of letters digits and underscore, and a total "
        "length of 3 to 16. Pattern: {p}",
        "Consuming the head separately means the tail quantifier counts 2 to 15, which puts the total "
        "in the right range. Pattern: {p}",
        "Bounding the tail at 3 to 16 would give a total length one too long. The head character has "
        "to be counted. Pattern: {p}",
    ],
    "uk_postcode": [
        "The outward code is variable length: one or two letters, a digit, then optionally another "
        "letter or digit. The inward code is fixed. Pattern: {p}",
        "Uppercase classes rather than a case-insensitive flag, since the rule specifies uppercase, "
        "and a single mandatory space between the two halves. Pattern: {p}",
        "Fixing the outward code at one letter would reject valid postcodes, and making the space "
        "optional would accept unspaced ones. Pattern: {p}",
    ],
    "float_signed": [
        "The sign is optional, and so is the integer part, so a bare fractional form is legal while a "
        "trailing dot with nothing after it is not. Pattern: {p}",
        "Making the digits-then-dot section optional and then requiring at least one digit handles "
        "both the leading-dot and trailing-dot cases at once. Pattern: {p}",
        "Requiring digits before the dot would reject a bare fraction; allowing the pattern to end at "
        "the dot would accept a trailing dot. Pattern: {p}",
    ],
    "port_number": [
        "The upper bound 65535 does not sit on a digit boundary, so I peel it off in descending ranges "
        "down to the unpadded low numbers, with a bare zero as its own branch. Pattern: {p}",
        "Range decomposition is the whole construction: each branch covers a block whose digits can be "
        "expressed with counted classes. Pattern: {p}",
        "Five loose digits would accept values well past the bound, and allowing leading zeros would "
        "admit padded ports. Pattern: {p}",
    ],
    "roman_numeral": [
        "Standard notation decomposes by magnitude, each group with its own subtractive pairs and "
        "repeat limits. Every group is optional, so a lookahead is needed to reject the empty string. "
        "Pattern: {p}",
        "Thousands, hundreds, tens, units, each an alternation; then a leading assertion that at least "
        "one numeral character is present. Pattern: {p}",
        "Because all four groups can match nothing, the pattern would otherwise accept the empty "
        "string, which the instruction rules out. Pattern: {p}",
    ],
    "latitude": [
        "The bound is 90, so it is special-cased: exactly 90 with optional zero decimals, or 0 to 89 "
        "with an optional fraction. An optional minus covers the south. Pattern: {p}",
        "Two branches split at the boundary, with the fractional part optional on each. Pattern: {p}",
        "Allowing a fraction on 90 without restricting it to zeros would accept values above the "
        "range. Pattern: {p}",
    ],
    "currency_usd": [
        "Grouped thousands are one to three digits followed by comma-and-three-digit groups. Cents, "
        "when present, are exactly two digits. Pattern: {p}",
        "Two branches, grouped and ungrouped, sharing the same optional two-digit fractional tail. "
        "Pattern: {p}",
        "A range on the decimals would accept one or three digits, which are not valid cents, and "
        "loose digits would accept mis-grouped thousands. Pattern: {p}",
    ],
    "jwt_format": [
        "A structural check only: three non-empty segments joined by literal dots, each drawn from the "
        "base64url alphabet. Pattern: {p}",
        "Two escaped dots give exactly three runs, and '+' on each run rejects empty segments. "
        "Pattern: {p}",
        "Using '*' on the segments would accept consecutive dots, and an unescaped dot would match any "
        "character. Pattern: {p}",
    ],
    "visa_card": [
        "The prefix is a literal 4, and two total lengths are valid. Since the prefix consumes one "
        "character, the remainder is 12 or 15 digits. Pattern: {p}",
        "One literal, then an alternation of two counted runs to give both exact lengths. Pattern: {p}",
        "A range quantifier would accept every length between the two valid ones, so the branches have "
        "to be explicit. Pattern: {p}",
    ],
    "amex_card": [
        "Two valid prefixes, 34 and 37, collapse into '3' followed by a class. The total length is "
        "exactly 15, so 13 digits remain. Pattern: {p}",
        "Shared leading digit plus a two-element class, then a counted run for the remainder. "
        "Pattern: {p}",
        "Writing the two prefixes as separate branches would duplicate the length count and invite an "
        "off-by-one. Pattern: {p}",
    ],
    "domain_name": [
        "Each label may contain letters, digits and hyphens but must not begin or end with one, so the "
        "label is alphanumeric, optional middle, alphanumeric. Pattern: {p}",
        "Repeated label-plus-dot groups, then a final alphabetic TLD with a minimum length. "
        "Pattern: {p}",
        "A flat class of letters, digits and hyphens would accept labels starting or ending with a "
        "hyphen, and would let a numeric TLD through. Pattern: {p}",
    ],
    "slug": [
        "Reading it as runs joined by single hyphens satisfies all three rules at once: no leading or "
        "trailing hyphen, and no doubled hyphen. Pattern: {p}",
        "An alphanumeric run, then any number of hyphen-plus-run groups. Lowercase only, so no "
        "uppercase in the class. Pattern: {p}",
        "Putting the hyphen inside one flat class would allow leading, trailing and repeated hyphens, "
        "all of which are forbidden. Pattern: {p}",
    ],
    "binary_multiple_of_four": [
        "Divisibility by four is structural in binary: the last two bits must be zero. Zero itself is "
        "the exception, written as a single digit. Pattern: {p}",
        "Two branches, the special case and a leading 1 with any middle and two trailing zeros. "
        "Pattern: {p}",
        "Without the special case, zero would be rejected; without the leading 1, padded forms would "
        "be accepted. Pattern: {p}",
    ],
    # ---------------------------------------------------------- extract
    "first_quoted": [
        "A greedy middle would run from the first quote to the last one in the text. A negated class "
        "stops at the very next quote, which is what 'first' means. Pattern: {p}",
        "Two literal quotes with a negated class between them, and the group inside so the delimiters "
        "stay out of the capture. Pattern: {p}",
        "Dot-star between quotes is the classic greedy mistake here: it spans every quoted run on the "
        "line instead of the first. Pattern: {p}",
    ],
    "email_domain": [
        "I anchor on the '@' but keep it outside the group, since only the domain is wanted. The "
        "alphabetic TLD is what stops the match running into trailing punctuation. Pattern: {p}",
        "Dot-separated labels ending in a letters-only final label, with the '@' consumed but not "
        "captured. Pattern: {p}",
        "Allowing any non-space character in the domain would swallow a trailing full stop into the "
        "capture. Bounding the last label to letters prevents that. Pattern: {p}",
    ],
    "http_status": [
        "The line holds several numbers, so anchoring matters more than the digit count. Matching the "
        "protocol and closing quote pins the position. Pattern: {p}",
        "Consume up to the quote, then capture exactly three digits, leaving the byte count outside "
        "the group. Pattern: {p}",
        "A bare three-digit match would fire on a path segment or the byte count. The protocol prefix "
        "is what disambiguates. Pattern: {p}",
    ],
    "file_extension": [
        "'Final dot' is the whole problem: a plain dot-then-anything stops at the first dot of a "
        "multi-part name. Pattern: {p}",
        "Anchoring to the end and forbidding dots inside the group forces the match onto the last dot. "
        "Pattern: {p}",
        "Without the end anchor the match would settle on the first dot, which is wrong for names with "
        "several. Pattern: {p}",
    ],
    "first_paren": [
        "Parentheses are metacharacters, so both need escaping. A greedy middle would span from the "
        "first opening bracket to the last closing one. Pattern: {p}",
        "Escaped brackets with a negated class between them, group inside. Pattern: {p}",
        "Forgetting to escape the brackets would create a group rather than match literal characters, "
        "and a dot-star middle would over-run. Pattern: {p}",
    ],
    "year_from_date": [
        "The match must cover the whole date so a stray four-digit number is not mistaken for a year, "
        "but only the year belongs in the group. Pattern: {p}",
        "Full ISO shape matched, capture group placed around the first field alone. Pattern: {p}",
        "Capturing four digits on their own would match any number of that length elsewhere in the "
        "text. Pattern: {p}",
    ],
    "first_hashtag": [
        "The hash is a marker, not part of the tag, so it stays outside the group. The word-character "
        "class ends the match at the first punctuation or space. Pattern: {p}",
        "Literal hash, then a captured run of word characters, which gives the boundary for free. "
        "Pattern: {p}",
        "Including the hash in the capture would return it as part of the tag name. Pattern: {p}",
    ],
    "first_mention": [
        "The thing to avoid is the '@' inside an email address. A mention only counts at the start or "
        "after whitespace. Pattern: {p}",
        "An alternation of a start anchor and a whitespace lookbehind asserts the position without "
        "consuming it, then the name is captured. Pattern: {p}",
        "Matching a bare '@' plus word characters would fire on the domain of any email address in the "
        "text. Pattern: {p}",
    ],
    "price_usd": [
        "The dollar sign anchors the match but is excluded from the group. Cents are optional and "
        "exactly two digits when present. Pattern: {p}",
        "Literal symbol outside the capture, digit run inside, optional two-digit fractional tail. "
        "Pattern: {p}",
        "A range on the decimals would accept one or three digits, which are not valid cents. "
        "Pattern: {p}",
    ],
    "query_param": [
        "Matching the bare key would also fire inside a longer parameter name, so the preceding "
        "character must be '?' or '&'. Pattern: {p}",
        "Separator class, literal key, then a negated class for the value so it stops at the next "
        "separator and an empty value still matches. Pattern: {p}",
        "Without the leading separator this would match the tail of a different parameter whose name "
        "ends in the same letters. Pattern: {p}",
    ],
    "html_tag_name": [
        "Tag names start with a letter and continue with letters and digits, which is exactly the "
        "boundary that stops the match before any attributes. Pattern: {p}",
        "Opening bracket, then a captured name class that excludes the slash, so a closing tag cannot "
        "match at its bracket. Pattern: {p}",
        "Capturing everything up to the closing bracket would drag the attributes in with the name. "
        "Pattern: {p}",
    ],
    "version_from_tag": [
        "The 'v' is a marker rather than part of the version, so it sits outside the capture group. "
        "Pattern: {p}",
        "Literal prefix, then the three dot-separated components captured together. Pattern: {p}",
        "Including the prefix in the group would return it as part of the version string. Pattern: {p}",
    ],
    "log_timestamp": [
        "Square brackets are metacharacters and need escaping. A greedy middle would run to the last "
        "closing bracket on the line. Pattern: {p}",
        "Escaped brackets around a negated class, with '*' rather than '+' so an empty pair still "
        "matches. Pattern: {p}",
        "An unescaped opening bracket would start a character class instead of matching a literal. "
        "Pattern: {p}",
    ],
    "acronym": [
        "An acronym is two or more uppercase letters, so a lone capital opening a sentence does not "
        "qualify. Pattern: {p}",
        "A counted uppercase run with word boundaries on both sides, so it is not pulled out of a "
        "mixed-case word. Pattern: {p}",
        "Without the minimum length every capitalised word would match its first letter; without the "
        "boundaries the run could start mid-word. Pattern: {p}",
    ],
    "srt_timecode": [
        "The arrow identifies which of the two timecodes on the line is the start one, so it is "
        "matched as a literal after the group. Pattern: {p}",
        "Fixed-width fields inside the capture, with a comma before the milliseconds as this format "
        "requires, then the arrow outside. Pattern: {p}",
        "Without the trailing arrow the match would settle on whichever timecode came first in the "
        "scan rather than the start one specifically. Pattern: {p}",
    ],
    "css_value": [
        "The property name anchors the match and the value runs to the semicolon. A negated class "
        "stops at the first one so a following declaration is not swallowed. Pattern: {p}",
        "Literal property, optional whitespace, captured value bounded by the terminator. Pattern: {p}",
        "A dot-star before the semicolon would run to the last one in the block, capturing several "
        "declarations at once. Pattern: {p}",
    ],
    "markdown_link_url": [
        "Two bracket types are involved and the square ones need escaping. The link text is skipped, "
        "the parenthesised URL captured. Pattern: {p}",
        "Negated classes on both the text and the URL keep the match inside the first link. "
        "Pattern: {p}",
        "Greedy middles here would span from the first link's opening bracket to the last link's "
        "closing parenthesis. Pattern: {p}",
    ],
    "ini_value": [
        "The key is a literal and the value runs to the end of its line. Optional spaces and tabs "
        "around the equals cover the formatting variants. Pattern: {p}",
        "Line-start anchor, literal key, flexible separator, then a captured remainder up to the line "
        "end. Pattern: {p}",
        "Without anchoring to the line start, a longer key ending in these letters would match. "
        "Pattern: {p}",
    ],
    "url_host": [
        "The scheme is matched but not captured, and the optional 'www.' is skipped by sitting in a "
        "non-capturing group ahead of the capture. Pattern: {p}",
        "Host ends at the first slash, question mark or whitespace, which a negated class expresses "
        "directly. Pattern: {p}",
        "Capturing the 'www.' would return it as part of the host, which the instruction excludes. "
        "Pattern: {p}",
    ],
    "sku_code": [
        "The literal prefix anchors the match and stays outside the group. The code is a fixed six "
        "characters. Pattern: {p}",
        "Counted uppercase-alphanumeric run inside the capture, with a trailing word boundary. "
        "Pattern: {p}",
        "Without the trailing boundary a longer code would match on its first six characters. "
        "Pattern: {p}",
    ],
    "block_between_markers": [
        "The content spans lines and the dot does not cross newlines by default, so this needs DOTALL. "
        "It also needs a lazy quantifier. Pattern: {p}",
        "Literal markers outside the group, lazy dot-star inside, DOTALL set so the middle can cross "
        "line breaks. Pattern: {p}",
        "A greedy quantifier would run from the first opening marker to the last closing one in the "
        "text, swallowing everything between two separate blocks. Pattern: {p}",
    ],
    "html_block_content": [
        "Two issues, both standard for multi-line blocks: DOTALL so the dot crosses newlines, and a "
        "lazy quantifier so the match stops at the first closing tag. Pattern: {p}",
        "Literal tags outside the capture, lazy middle inside, DOTALL enabled. Pattern: {p}",
        "Greedy matching here would join the first opening tag to the last closing one in the "
        "document. Pattern: {p}",
    ],
    "mail_header_value": [
        "Three things combine: MULTILINE because the header can be on any line, IGNORECASE because the "
        "name may be written in any case, and a line-end anchor for the value. Pattern: {p}",
        "Line-anchored literal name, optional spaces consumed outside the group, value captured to the "
        "line end, with both flags set. Pattern: {p}",
        "Without MULTILINE the anchors would only match the whole string; without IGNORECASE an "
        "uppercased header name would be missed. Pattern: {p}",
    ],
    # ---------------------------------------------------------- substitute
    "comma_decimal": [
        "A substitution has to rebuild the number rather than delete the comma, so the digit runs need "
        "capture groups. Pattern: {p} with replacement {r}",
        "Two groups, one per digit run, reassembled around a literal dot. Pattern: {p} with "
        "replacement {r}",
        "Replacing the comma alone with no groups would delete the surrounding digits along with it. "
        "Pattern: {p} with replacement {r}",
    ],
    "collapse_whitespace": [
        "The quantifier belongs on the whitespace class so an entire run is consumed by one match. "
        "Pattern: {p} with replacement {r}",
        "One match per run, replaced wholesale by a single space, which is what makes runs of any "
        "length collapse correctly. Pattern: {p} with replacement {r}",
        "Matching a single whitespace character would replace each one individually and leave the run "
        "the same length. Pattern: {p} with replacement {r}",
    ],
    "strip_trailing_ws": [
        "An end-of-string anchor alone would only clean the final line, so this needs MULTILINE. "
        "Pattern: {p}",
        "Spaces and tabs only, anchored to the line end, so the breaks themselves survive. Pattern: {p}",
        "Using the full whitespace class would consume the line breaks as well as the trailing spaces. "
        "Pattern: {p}",
    ],
    "mask_card": [
        "The rule is positional: a digit is masked if enough digits follow it. A lookahead expresses "
        "that without consuming what it counts. Pattern: {p} with replacement {r}",
        "Single digit matched, lookahead counting the tail, so each digit is judged independently. "
        "Pattern: {p} with replacement {r}",
        "Consuming the following digits instead of asserting them would mask them too, destroying the "
        "tail that has to survive. Pattern: {p} with replacement {r}",
    ],
    "mask_ssn": [
        "The last group must survive, so it is captured and written back while the first two are "
        "replaced by literal asterisks. Pattern: {p} with replacement {r}",
        "Only the final field needs a group; everything before it is matched and discarded. "
        "Pattern: {p} with replacement {r}",
        "Replacing the whole match with asterisks would destroy the last four digits the instruction "
        "says to keep. Pattern: {p} with replacement {r}",
    ],
    "swap_name": [
        "Both names go into groups so the replacement can reorder them. Pattern: {p} with "
        "replacement {r}",
        "The capitalisation rule keeps the match on proper names: uppercase initial then lowercase, on "
        "both sides of the space. Pattern: {p} with replacement {r}",
        "A generic word class would match any two adjacent words, including ordinary lowercase prose. "
        "Pattern: {p} with replacement {r}",
    ],
    "csv_swap_columns": [
        "Each line is independent, so MULTILINE makes the anchors line boundaries. Pattern: {p} with "
        "replacement {r}",
        "Excluding the comma from the field class stops one field swallowing the delimiter; the two "
        "groups are written back reversed. Pattern: {p} with replacement {r}",
        "A dot-star field would consume the comma and the second field along with it, leaving nothing "
        "to swap. Pattern: {p} with replacement {r}",
    ],
    "date_us_to_iso": [
        "Three fixed-width fields into three groups, emitted in a different order with hyphens. "
        "Pattern: {p} with replacement {r}",
        "Nothing needs validating here, only reordering, so counted digit runs suffice. Pattern: {p} "
        "with replacement {r}",
        "Getting the group order wrong in the replacement is the whole risk: the day and month must "
        "not swap. Pattern: {p} with replacement {r}",
    ],
    "camel_underscore": [
        "The separator is inserted between two characters that both have to survive, so both go into "
        "capture groups and the replacement writes them back around it. Pattern: {p} with "
        "replacement {r}",
        "Requiring a lowercase letter or digit on the left means the pattern cannot fire at the start "
        "of a name that already begins with a capital. Pattern: {p} with replacement {r}",
        "Matching the uppercase letter alone and replacing it would delete it: the insert has to "
        "happen without consuming either side, so both are captured. Pattern: {p} with "
        "replacement {r}",
    ],
    "strip_html_tags": [
        "A tag runs from an opening bracket to the next closing one, so a negated class rather than a "
        "greedy dot-star. Pattern: {p}",
        "Negated class between the literal brackets, replaced with nothing. Pattern: {p}",
        "The greedy version would match from the first '<' to the last '>' and delete all the text in "
        "between. Pattern: {p}",
    ],
    "md_header_to_html": [
        "Headers sit on their own lines, so MULTILINE makes the anchors line-relative. Pattern: {p} "
        "with replacement {r}",
        "The heading text is captured so the replacement can wrap it, and the required space stops a "
        "deeper header matching. Pattern: {p} with replacement {r}",
        "Without the space after the hash, a level-two header would match as a level-one one. "
        "Pattern: {p} with replacement {r}",
    ],
    "crlf_to_lf": [
        "A literal two-character sequence replaced by one. Pattern: {p}",
        "Matching the carriage return explicitly as part of the pair means a line already ending in a "
        "bare newline is untouched. Pattern: {p}",
        "Replacing the carriage return alone would work here too, but matching the pair makes the "
        "intent explicit and cannot affect a stray return. Pattern: {p}",
    ],
    "pad_single_digits": [
        "The digit must be standalone, so word boundaries are needed on both sides. Pattern: {p} with "
        "replacement {r}",
        "Captured digit written back after an added zero, bounded so longer numbers are untouched. "
        "Pattern: {p} with replacement {r}",
        "Without the boundaries the first digit of a longer number would be padded, corrupting it. "
        "Pattern: {p} with replacement {r}",
    ],
    "comma_to_tab": [
        "A plain literal replacement. Pattern: {p} with replacement {r}",
        "The only care needed is on the replacement side, where the tab is written as an escape. "
        "Pattern: {p} with replacement {r}",
        "No groups or classes are involved; a single literal character maps to another. Pattern: {p} "
        "with replacement {r}",
    ],
    "remove_non_ascii": [
        "ASCII is the code points up to 127, so the class negates that range. Pattern: {p}",
        "Negated range matched one character at a time and replaced with nothing. Pattern: {p}",
        "Listing the characters to remove would be impossible; negating the range that stays is the "
        "tractable direction. Pattern: {p}",
    ],
    "normalize_phone": [
        "Three fixed-width groups captured and rejoined with hyphens. Pattern: {p} with "
        "replacement {r}",
        "Word boundaries keep the match from starting partway into a longer digit run. Pattern: {p} "
        "with replacement {r}",
        "Without the boundaries an eleven-digit number would have its first ten formatted, leaving a "
        "stray digit. Pattern: {p} with replacement {r}",
    ],
    "replace_domain": [
        "The host is a literal, but its dots must be escaped or they would match any character. "
        "Pattern: {p}",
        "Escaped literal matched, plain literal substituted, no groups involved. Pattern: {p}",
        "Unescaped dots would let the pattern fire on a similar-looking host with different separator "
        "characters. Pattern: {p}",
    ],
    "relative_to_absolute": [
        "Only the path varies, so it is captured while the attribute syntax around it is matched and "
        "rewritten. Pattern: {p} with replacement {r}",
        "Requiring the leading slash inside the group limits this to root-relative links. Pattern: {p} "
        "with replacement {r}",
        "Without the required slash this would also rewrite absolute URLs, which must be left alone. "
        "Pattern: {p} with replacement {r}",
    ],
    "collapse_blank_lines": [
        "The quantifier lower bound is the entire rule: three or more collapse, fewer stay. "
        "Pattern: {p} with replacement {r}",
        "The whole run is matched in one go and replaced by two newlines. Pattern: {p} with "
        "replacement {r}",
        "A lower bound of two would collapse ordinary paragraph breaks, which must survive. "
        "Pattern: {p} with replacement {r}",
    ],
    "strip_code_comments": [
        "The comment runs to the end of its line, so the class excludes the newline rather than using "
        "a dot. Pattern: {p}",
        "Preceding spaces and tabs are consumed as part of the match so no gap is left behind. "
        "Pattern: {p}",
        "Using a dot-star would stop at the line end anyway, but an explicit newline exclusion makes "
        "it safe if DOTALL is ever set. Pattern: {p}",
    ],
    "strip_block_comments": [
        "Block comments span lines, so the dot needs DOTALL, and the quantifier has to be lazy. The "
        "asterisks are metacharacters and need escaping. Pattern: {p}",
        "Escaped delimiters with a lazy dot-star between them, DOTALL enabled. Pattern: {p}",
        "A greedy quantifier would delete everything between the first opener and the last closer, "
        "taking the real code with it. Pattern: {p}",
    ],
    "redact_secret_block": [
        "Unlike an extraction, the markers are being replaced too, so they sit inside the match rather "
        "than outside it. Pattern: {p} with replacement {r}",
        "Lazy middle with DOTALL so the content can cross lines, and both markers consumed. "
        "Pattern: {p} with replacement {r}",
        "A greedy quantifier would merge two separate blocks into one redaction, destroying the text "
        "between them. Pattern: {p} with replacement {r}",
    ],
    "normalize_header_case": [
        "Two flags together: IGNORECASE so any spelling matches, and MULTILINE so the anchor means "
        "line start. Pattern: {p} with replacement {r}",
        "Only the header name is matched, so the value after the colon is untouched. Pattern: {p} with "
        "replacement {r}",
        "Without the line-start anchor a longer header name ending in these letters would match. "
        "Pattern: {p} with replacement {r}",
    ],
}


# Keyed by (arc concept, turn index), 3 phrasings each. Turns 2 and 3 reason
# from the PREVIOUS ANSWER rather than from previous reasoning, because the
# Thinking model's chat template strips <think> out of history -- by the next
# turn the earlier trace is gone and only the pattern remains in context.
ARC_TRACES = {
    ("comma_decimal_chain", 1): [
        "I need to locate the number before transforming it, so this turn is just a capture: a digit "
        "run, a comma, another digit run, all in one group. Pattern: {p}",
        "Capture first, transform later. One group spanning both digit runs and the comma. Pattern: {p}",
        "Nothing is being rewritten yet, so no separate groups are needed around each run. Pattern: {p}",
    ],
    ("comma_decimal_chain", 2): [
        "The previous pattern had no capture groups, so a replacement has nothing to write back. "
        "Restructuring into two groups is required. Pattern: {p} with replacement {r}",
        "This is not an extension of the last answer but a rebuild: one group per digit run, "
        "reassembled around a dot. Pattern: {p} with replacement {r}",
        "Appending a replacement to the single-group pattern would delete the digits, since they were "
        "inside the match but not captured separately. Pattern: {p} with replacement {r}",
    ],
    ("comma_decimal_chain", 3): [
        "Thousands separators have three digits after the comma, decimals one or two. Bounding the "
        "second group is not enough alone, so a negative lookahead forbids a further digit. "
        "Pattern: {p} with replacement {r}",
        "Narrowing the second group to one or two digits still lets it match the first two of a "
        "longer group, hence the lookahead. Pattern: {p} with replacement {r}",
        "Without the lookahead a grouped thousand would have its first two digits converted, "
        "corrupting the number. Pattern: {p} with replacement {r}",
    ],
    ("card_mask_chain", 1): [
        "Sixteen digits total and I want the final four, so counting off the leading twelve outside "
        "the group puts the capture in the right place. Pattern: {p}",
        "Positional counting rather than anchoring: twelve consumed, four captured. Pattern: {p}",
        "Anchoring to the end would work on a bare number but not inside surrounding text; the counted "
        "prefix is more robust. Pattern: {p}",
    ],
    ("card_mask_chain", 2): [
        "The task turns positional: mask a digit if four or more follow it. A lookahead tests that "
        "without consuming what it counts. Pattern: {p} with replacement {r}",
        "One digit matched at a time, with the tail asserted rather than consumed, so the last four "
        "survive. Pattern: {p} with replacement {r}",
        "Consuming the trailing digits instead of asserting them would mask them too. Pattern: {p} "
        "with replacement {r}",
    ],
    ("card_mask_chain", 3): [
        "This reverts to the capture task from the first turn rather than building on the masking: "
        "same shape, but the group moves to the front. Pattern: {p}",
        "Going back to extraction, with the counted twelve now trailing the captured four. "
        "Pattern: {p}",
        "The masking answer is discarded entirely here; only the original counting construction is "
        "reused, mirrored. Pattern: {p}",
    ],
    ("phone_chain", 1): [
        "Ten digits, no separators. Under fullmatch a single counted quantifier gives the exact "
        "length. Pattern: {p}",
        "One counted run, no anchors needed. Pattern: {p}",
        "Using '+' would accept any length of digit run, which the fixed length rules out. "
        "Pattern: {p}",
    ],
    ("phone_chain", 2): [
        "Formatting inserts separators without reordering, so each of the three fields needs its own "
        "group. Pattern: {p} with replacement {r}",
        "Three counted groups rejoined with hyphens, with word boundaries so the match cannot start "
        "partway into a longer run. Pattern: {p} with replacement {r}",
        "Without boundaries an eleven-digit number would have its first ten formatted and a stray "
        "digit left over. Pattern: {p} with replacement {r}",
    ],
    ("phone_chain", 3): [
        "The country code must survive when present and cost nothing when absent, so it goes in its "
        "own optional group referenced first. Pattern: {p} with replacement {r}",
        "Python substitutes an unmatched optional group as empty, so the same replacement handles both "
        "the prefixed and bare forms. Pattern: {p} with replacement {r}",
        "Making the prefix mandatory would break the plain ten-digit case that already worked. "
        "Pattern: {p} with replacement {r}",
    ],
    ("us_date_chain", 1): [
        "Month and day both need explicit range branches rather than two loose digits, and every "
        "branch is two characters so the padding is structural. Pattern: {p}",
        "Range alternations for the two bounded fields, a plain counted run for the year. Pattern: {p}",
        "Two loose digits per field would accept an impossible month or day. Pattern: {p}",
    ],
    ("us_date_chain", 2): [
        "Validation is done; this turn only rearranges. Three groups, emitted year first with hyphens. "
        "Pattern: {p} with replacement {r}",
        "No range checking is needed for a pure reordering, so counted runs suffice. Pattern: {p} with "
        "replacement {r}",
        "Carrying the range branches forward would work but adds nothing here, since the input is "
        "already known to be valid. Pattern: {p} with replacement {r}",
    ],
    ("us_date_chain", 3): [
        "The pattern shape is unchanged because the input still has three slash-separated fields; only "
        "the order differs, so the fix is entirely in the replacement. Pattern: {p} with "
        "replacement {r}",
        "Same match, different field order: the day now comes from the first group and the month from "
        "the second. Pattern: {p} with replacement {r}",
        "Changing the pattern here would be the wrong instinct; the input format is identical and only "
        "the interpretation changed. Pattern: {p} with replacement {r}",
    ],
    ("semver_chain", 1): [
        "Exactly three dot-separated numeric components, so three digit runs joined by escaped dots. "
        "Pattern: {p}",
        "Component count is the only rule in play yet, so plain digit runs are enough. Pattern: {p}",
        "Unescaped dots would match any character, letting malformed versions through. Pattern: {p}",
    ],
    ("semver_chain", 2): [
        "An optional suffix means a non-capturing group ending in '?', introduced by a literal hyphen. "
        "Pattern: {p}",
        "The base stays as it was; only an optional tail is appended, with '+' inside so a bare "
        "trailing hyphen is rejected. Pattern: {p}",
        "Using '*' inside the suffix group would accept a hyphen with nothing after it. Pattern: {p}",
    ],
    ("semver_chain", 3): [
        "The reported failure is a padded component being accepted, because a plain digit run does not "
        "care about a leading zero. Pattern: {p}",
        "Each component becomes either exactly zero or a nonzero digit followed by anything, which "
        "keeps a bare zero legal while rejecting padding. Pattern: {p}",
        "Simply forbidding a zero at the start of a component would also reject a legitimate zero "
        "component, so the alternation is needed. Pattern: {p}",
    ],
    ("hex_color_chain", 1): [
        "Two valid lengths, so an alternation rather than a range which would admit four and five. "
        "Pattern: {p}",
        "Explicit counted branches sharing the hex class. Pattern: {p}",
        "A range quantifier from three to six is the tempting shortcut and it is wrong. Pattern: {p}",
    ],
    ("hex_color_chain", 2): [
        "The alpha forms add four and eight. Three and four are adjacent so they collapse into a "
        "range, while six and eight stay separate because seven is still invalid. Pattern: {p}",
        "Two of the four lengths are contiguous and two are not, so the alternation mixes a range with "
        "explicit counts. Pattern: {p}",
        "Widening the whole thing to three through eight would accept five and seven digits. "
        "Pattern: {p}",
    ],
    ("hex_color_chain", 3): [
        "This goes back to the first turn's three-or-six rule, discarding the alpha branches, then "
        "applies one new restriction: the alphabet drops uppercase. Pattern: {p}",
        "Reverting the length rule and narrowing the class in the same step. Pattern: {p}",
        "Keeping the four-and-eight branches would contradict the instruction to return to the "
        "original rule. Pattern: {p}",
    ],
    ("time_chain", 1): [
        "Hours split into two branches and minutes into one, each field exactly two characters so the "
        "padding is enforced structurally. Pattern: {p}",
        "Standard range decomposition for a bounded two-field time. Pattern: {p}",
        "Loose digits would accept hours past 23 and minutes past 59. Pattern: {p}",
    ],
    ("time_chain", 2): [
        "Seconds reuse the same construction as minutes, wrapped in an optional group so both forms "
        "are accepted. Pattern: {p}",
        "One optional non-capturing group appended; nothing else changes. Pattern: {p}",
        "Making the seconds mandatory would reject the short form that already worked. Pattern: {p}",
    ],
    ("time_chain", 3): [
        "Now I want the complement of the hour rule. Hours of 24 and above means '2' with 4-9, or a "
        "leading digit of 3-9 with any second digit. Pattern: {p}",
        "Only the hour is inverted; the minute range stays valid, since the instruction still "
        "describes a well-formed minute field. Pattern: {p}",
        "Negating the whole pattern would also admit malformed minutes, which is not what is being "
        "asked for. Pattern: {p}",
    ],
    ("log_level_chain", 1): [
        "Five fixed names, so a literal alternation inside the group with word boundaries either side. "
        "Pattern: {p}",
        "Alternation of literals, bounded so a level name cannot match inside a longer word. "
        "Pattern: {p}",
        "Without the boundaries a level name would match as a substring of an unrelated identifier. "
        "Pattern: {p}",
    ],
    ("log_level_chain", 2): [
        "Same structure, smaller alternation. Dropping the lower severities means a line carrying only "
        "those yields no match. Pattern: {p}",
        "Only the branch list narrows; the group and boundaries are unchanged. Pattern: {p}",
        "Keeping the other names and filtering afterwards is not an option here, since the pattern "
        "itself has to reject them. Pattern: {p}",
    ],
    ("log_level_chain", 3): [
        "One more branch is added back into the alternation. Everything else carries over unchanged. "
        "Pattern: {p}",
        "Composing an extra case into the existing branch list. Pattern: {p}",
        "Rebuilding from the original five would over-shoot, since only one of the three dropped names "
        "is being restored. Pattern: {p}",
    ],
    ("port_chain", 1): [
        "The bound does not sit on a digit boundary, so I peel it off in descending ranges down to the "
        "unpadded low numbers, with a bare zero as its own branch. Pattern: {p}",
        "Range decomposition into blocks that counted classes can express. Pattern: {p}",
        "Five loose digits would accept values well past the bound, and padding would let zero-prefixed "
        "ports through. Pattern: {p}",
    ],
    ("port_chain", 2): [
        "Narrowing the upper boundary changes how the range decomposes: a top block, then everything "
        "below as a nonzero digit with up to two more. Zero is excluded now. Pattern: {p}",
        "Two branches replace the previous six, and the standalone zero branch goes. Pattern: {p}",
        "Keeping the old decomposition and adding a bound is not possible in one pattern; the "
        "boundary has to be rebuilt. Pattern: {p}",
    ],
    ("port_chain", 3): [
        "This restores the full range from the first turn rather than extending the privileged one. "
        "The only change is dropping the standalone zero branch. Pattern: {p}",
        "Reverting to the original decomposition, minus the zero case, since every remaining branch "
        "starts with a nonzero digit. Pattern: {p}",
        "Extending the narrowed pattern upward would be the wrong move; the instruction points back to "
        "the earlier answer. Pattern: {p}",
    ],
    ("email_domain_chain", 1): [
        "A structural check: two runs of non-whitespace non-@ characters joined by an '@', with a "
        "literal dot forcing the domain to have one. Pattern: {p}",
        "Negated classes for the parts, literal separators between them. Pattern: {p}",
        "Dot-star parts would swallow whitespace and stray '@' characters. Pattern: {p}",
    ],
    ("email_domain_chain", 2): [
        "The domain stops being a wildcard and becomes a literal, whose dot must be escaped. "
        "Pattern: {p}",
        "Only the right-hand side changes; the local part keeps its negated class. Pattern: {p}",
        "An unescaped dot in the literal domain would match any character and let a lookalike through. "
        "Pattern: {p}",
    ],
    ("email_domain_chain", 3): [
        "The pattern text does not change at all; what changes is how it is applied. IGNORECASE makes "
        "the literal domain match in any letter case. Pattern: {p}",
        "A flag rather than a pattern edit is the right tool: rewriting the domain as character "
        "classes would be verbose and easy to get wrong. Pattern: {p}",
        "Adding uppercase alternatives by hand would double the literal's length for no benefit. "
        "Pattern: {p}",
    ],
    ("whitespace_chain", 1): [
        "The quantifier belongs on the whitespace class so an entire run is consumed by one match. "
        "Pattern: {p} with replacement {r}",
        "One match per run, replaced wholesale by a single space. Pattern: {p} with replacement {r}",
        "Matching one whitespace character at a time would replace each individually and leave the run "
        "unchanged in length. Pattern: {p} with replacement {r}",
    ],
    ("whitespace_chain", 2): [
        "The reported failure is line breaks being eaten, because the whitespace class includes the "
        "newline. Narrowing to spaces and tabs fixes it. Pattern: {p} with replacement {r}",
        "Only the class changes; the quantifier and replacement stay as they were. Pattern: {p} with "
        "replacement {r}",
        "Excluding the newline by negation would be more fragile than just listing the two characters "
        "that should collapse. Pattern: {p} with replacement {r}",
    ],
    ("whitespace_chain", 3): [
        "This is no longer a collapse but a deletion at a position, so the pattern needs an end-of-line "
        "anchor, which only means end-of-line under MULTILINE. Pattern: {p}",
        "Different operation, different construction: anchor plus deletion rather than run plus "
        "replacement. Pattern: {p}",
        "Without the flag only the final line would be cleaned, since the anchor would mean "
        "end-of-string. Pattern: {p}",
    ],
    ("csv_chain", 1): [
        "Each row is independent, so MULTILINE turns the anchors into line boundaries. Pattern: {p} "
        "with replacement {r}",
        "Excluding the comma from the field class keeps a field from swallowing the delimiter. "
        "Pattern: {p} with replacement {r}",
        "A dot-star field would consume the comma and the second field along with it. Pattern: {p} "
        "with replacement {r}",
    ],
    ("csv_chain", 2): [
        "The failure is a three-field row not matching, because the previous pattern anchored a second "
        "field directly to the line end. Pattern: {p} with replacement {r}",
        "Adding a third field and group makes it match, and the replacement writes all three back "
        "reversed. Pattern: {p} with replacement {r}",
        "Making the third field optional would leave the replacement referring to a group that may not "
        "exist. Pattern: {p} with replacement {r}",
    ],
    ("csv_chain", 3): [
        "Fixing the field count is the wrong shape when rows vary. Dropping the end anchor generalises "
        "it to the first two fields wherever the row ends. Pattern: {p} with replacement {r}",
        "The match now covers only the leading pair and the rest of the line is left untouched. "
        "Pattern: {p} with replacement {r}",
        "Enumerating four, five and six field variants would not generalise; removing the anchor does. "
        "Pattern: {p} with replacement {r}",
    ],
    ("comment_chain", 1): [
        "The comment runs to the end of its line, so the class excludes newlines rather than using a "
        "dot, keeping the break in place. Pattern: {p}",
        "Preceding spaces are consumed as part of the match so no gap is left behind. Pattern: {p}",
        "A dot-star would stop at the line end anyway, but the explicit exclusion is safe even if "
        "DOTALL is set later. Pattern: {p}",
    ],
    ("comment_chain", 2): [
        "The failure is an empty line left where a whole-line comment was. Anchoring to line start "
        "under MULTILINE restricts the match to full-line comments. Pattern: {p}",
        "Consuming the trailing newline as well is what removes the blank line rather than leaving it. "
        "Pattern: {p}",
        "Without the anchor this would still strip inline comments, which is the behaviour being "
        "narrowed away. Pattern: {p}",
    ],
    ("comment_chain", 3): [
        "This returns to the first turn's trailing-comment behaviour rather than the whole-line rule, "
        "with one addition: a negative lookahead so a shebang is left alone. Pattern: {p}",
        "Reverting the anchor and the newline consumption, then excluding one specific two-character "
        "opening. Pattern: {p}",
        "Keeping the line-start anchor would contradict the instruction to go back to the original "
        "behaviour. Pattern: {p}",
    ],
    ("url_chain", 1): [
        "The scheme is matched but not captured, and the optional prefix sits in a non-capturing group "
        "ahead of the capture so it is skipped. Pattern: {p}",
        "The host ends at the first slash, question mark or whitespace, which a negated class gives "
        "directly. Pattern: {p}",
        "Capturing the leading prefix would return it as part of the host. Pattern: {p}",
    ],
    ("url_chain", 2): [
        "Same anchoring work, different field captured: the host is now consumed outside the group and "
        "the path goes inside it. Pattern: {p}",
        "The group moves rather than the pattern being rebuilt, running from the leading slash up to "
        "any query string. Pattern: {p}",
        "Capturing from the scheme onward would include the host in the path. Pattern: {p}",
    ],
    ("url_chain", 3): [
        "Nothing needs capturing: the host is a fixed literal being swapped for another, with its dots "
        "escaped so a similar host cannot match. Pattern: {p}",
        "Scheme and path fall outside the match entirely, so they survive untouched. Pattern: {p}",
        "Matching the whole URL and rebuilding it would need groups for parts that never change. "
        "Pattern: {p}",
    ],
    ("ssn_chain", 1): [
        "The last group must survive, so it is captured and written back while the first two are "
        "replaced by literal asterisks. Pattern: {p} with replacement {r}",
        "Only the final field needs a group; everything before it is matched and discarded. "
        "Pattern: {p} with replacement {r}",
        "Replacing the whole match would destroy the digits the instruction says to keep. Pattern: {p} "
        "with replacement {r}",
    ],
    ("ssn_chain", 2): [
        "Without hyphens there are no separators to anchor on, so the split is purely positional: five "
        "consumed, four captured. Pattern: {p} with replacement {r}",
        "Word boundaries keep the match from starting partway into a longer digit run. Pattern: {p} "
        "with replacement {r}",
        "Carrying the hyphenated pattern forward would simply not match this format at all. "
        "Pattern: {p} with replacement {r}",
    ],
    ("ssn_chain", 3): [
        "Nine consecutive digits describe plenty of things that are not SSNs, which is why phone "
        "numbers were hit. Requiring the label immediately before fixes it. Pattern: {p} with "
        "replacement {r}",
        "Putting the label requirement in a lookbehind keeps it out of the replacement, so it is not "
        "rewritten. Pattern: {p} with replacement {r}",
        "Consuming the label instead of asserting it would mean rebuilding it in the replacement, "
        "which is avoidable. Pattern: {p} with replacement {r}",
    ],
    ("markdown_chain", 1): [
        "The instruction restricts this to the header at the very start, so the anchor should mean "
        "start-of-string, which is its default with no flag. Pattern: {p} with replacement {r}",
        "Line-start anchor without MULTILINE, captured heading text, wrapped by the replacement. "
        "Pattern: {p} with replacement {r}",
        "Adding MULTILINE here would convert headers further down the document, which is out of scope "
        "for this turn. Pattern: {p} with replacement {r}",
    ],
    ("markdown_chain", 2): [
        "Only the marker changes: two hashes instead of one. The required space still prevents a "
        "deeper header matching. Pattern: {p} with replacement {r}",
        "Same construction retargeted at a different heading level. Pattern: {p} with replacement {r}",
        "Making the second hash optional would match both levels, which is not what was asked. "
        "Pattern: {p} with replacement {r}",
    ],
    ("markdown_chain", 3): [
        "Scope changes here, not the marker. MULTILINE redefines the anchors as line boundaries so "
        "every matching line is converted. Pattern: {p} with replacement {r}",
        "The pattern gains an end-of-line anchor and the flag; the marker and capture are unchanged. "
        "Pattern: {p} with replacement {r}",
        "Trying to match repeated headers with a quantifier would not work, since they are separated "
        "by arbitrary body text. Pattern: {p} with replacement {r}",
    ],
    ("acronym_chain", 1): [
        "A run of two or more uppercase letters, bounded so it is not pulled from the middle of a "
        "mixed-case word. Pattern: {p}",
        "Counted uppercase class with word boundaries either side. Pattern: {p}",
        "Without a minimum length every capitalised word would match its initial. Pattern: {p}",
    ],
    ("acronym_chain", 2): [
        "The same construction with the case class flipped and the minimum length raised. Pattern: {p}",
        "Retargeting to lowercase words: same boundaries, different class and count. Pattern: {p}",
        "Keeping the uppercase class and adding a lowercase branch would match either, which is a "
        "later turn's job, not this one. Pattern: {p}",
    ],
    ("acronym_chain", 3): [
        "Both rules now apply and whichever matches earliest wins, which is what an alternation inside "
        "one capture group gives. Pattern: {p}",
        "The shared word boundaries stay outside the alternation so both branches inherit them. "
        "Pattern: {p}",
        "Two separate patterns would need combining afterwards; a single alternation lets the engine "
        "pick the earlier match. Pattern: {p}",
    ],
    ("trailing_ws_chain", 1): [
        "With no flag the end anchor means end-of-string, which matches the instruction to clean only "
        "the very end of the text. Pattern: {p}",
        "Spaces and tabs only, so line breaks survive. Pattern: {p}",
        "Adding MULTILINE here would clean every line, which this turn explicitly does not want. "
        "Pattern: {p}",
    ],
    ("trailing_ws_chain", 2): [
        "The pattern is already right; only its scope is wrong. MULTILINE redefines the anchor as "
        "end-of-line. Pattern: {p}",
        "A flag change rather than a pattern change. Pattern: {p}",
        "Rewriting the pattern to look for a newline would work but is unnecessary when the flag says "
        "exactly this. Pattern: {p}",
    ],
    ("trailing_ws_chain", 3): [
        "Switching from rewriting to capturing means the line content needs a group, and the "
        "quantifier must be lazy. Pattern: {p}",
        "A greedy quantifier would swallow the trailing whitespace into the capture before the "
        "optional class could take it. Pattern: {p}",
        "The anchors and flag carry over; what is new is the capture group and its laziness. "
        "Pattern: {p}",
    ],
    ("ini_value_chain", 1): [
        "A single line, so the default anchors are correct. Optional spaces around the equals cover "
        "the formatting variants. Pattern: {p}",
        "Literal key, flexible separator, value captured to the end. Pattern: {p}",
        "Requiring exactly one space around the equals would miss the unspaced and over-spaced forms. "
        "Pattern: {p}",
    ],
    ("ini_value_chain", 2): [
        "The key can now be on any line, which is exactly what MULTILINE changes about the anchors. "
        "Pattern: {p}",
        "Anchoring to line start is also what stops a longer key ending in this one from matching. "
        "Pattern: {p}",
        "Dropping the anchors entirely would match the key as a substring of a different setting. "
        "Pattern: {p}",
    ],
    ("ini_value_chain", 3): [
        "Flags compose, so this keeps MULTILINE and adds IGNORECASE. The pattern text is unchanged. "
        "Pattern: {p}",
        "Only the case sensitivity of the literal key differs from the previous answer. Pattern: {p}",
        "Writing the key as a class of upper and lower alternatives would be verbose and easy to get "
        "wrong compared with a flag. Pattern: {p}",
    ],
    ("blank_line_chain", 1): [
        "The lower bound on the quantifier is the entire rule: three or more collapse, fewer are left "
        "alone. Pattern: {p} with replacement {r}",
        "The whole run is matched at once and replaced wholesale. Pattern: {p} with replacement {r}",
        "A bound of two would collapse ordinary paragraph breaks, which must survive. Pattern: {p} "
        "with replacement {r}",
    ],
    ("blank_line_chain", 2): [
        "A different operation on different material: whole lines of only spaces or tabs, deleted with "
        "their break. Anchoring to line start needs MULTILINE. Pattern: {p}",
        "This is not a narrowing of the previous rule but a separate one, so the construction changes "
        "entirely. Pattern: {p}",
        "Adjusting the newline quantifier would not help, since the lines here contain whitespace "
        "rather than being empty. Pattern: {p}",
    ],
    ("blank_line_chain", 3): [
        "Back to the newline-collapsing shape from the first turn rather than the whitespace-line "
        "deletion, with the bound dropped to two. Pattern: {p} with replacement {r}",
        "Reverting the construction and tightening the bound in one step. Pattern: {p} with "
        "replacement {r}",
        "Keeping the whitespace-line pattern would not collapse runs of bare newlines at all. "
        "Pattern: {p} with replacement {r}",
    ],
    ("password_chain", 1): [
        "Three conditions must all hold, which is what lookaheads are for: each scans from the start "
        "without consuming. Pattern: {p}",
        "Assertions first, then a restricted alphabet with a minimum length bound. Pattern: {p}",
        "Consuming the required characters in order would fail, since they can appear in any order. "
        "Pattern: {p}",
    ],
    ("password_chain", 2): [
        "This needs two coordinated changes, not one: a fourth lookahead, and the consuming class "
        "widened to permit those characters. Pattern: {p}",
        "Composing an extra requirement means touching both the assertion list and the alphabet. "
        "Pattern: {p}",
        "Adding only the assertion would make every valid password unmatchable, since the required "
        "character could not be consumed. Pattern: {p}",
    ],
    ("password_chain", 3): [
        "Only the length bound moves. All four assertions and the alphabet stay as they were. "
        "Pattern: {p}",
        "A single number changes; nothing else about the previous answer is affected. Pattern: {p}",
        "Relaxing one of the lookaheads instead would change which passwords qualify, not how long "
        "they must be. Pattern: {p}",
    ],
    ("currency_chain", 1): [
        "Grouped thousands are one to three digits followed by comma-and-three-digit groups, with "
        "exactly two decimals when present. Pattern: {p}",
        "Two branches, grouped and ungrouped, sharing an optional fractional tail. Pattern: {p}",
        "A range on the decimals would accept one or three digits, which are not valid cents. "
        "Pattern: {p}",
    ],
    ("currency_chain", 2): [
        "The two symbols go into a class at the front, and I factor the number into one group so the "
        "alternation applies to the amount. Pattern: {p}",
        "Composing a second currency without duplicating the whole amount grammar on each branch. "
        "Pattern: {p}",
        "Writing two full alternatives, one per symbol, would double the pattern and invite the two "
        "copies to drift apart. Pattern: {p}",
    ],
    ("currency_chain", 3): [
        "Making the decimal part mandatory just means lifting it out of its optional group. "
        "Pattern: {p}",
        "The grouped and ungrouped integer branches stay; only the optionality of the cents changes. "
        "Pattern: {p}",
        "Adding a separate branch requiring decimals would leave the old optional one still matching. "
        "Pattern: {p}",
    ],
    ("slug_chain", 1): [
        "Reading it as runs joined by single separators satisfies all three rules at once. Pattern: {p}",
        "A run, then any number of separator-plus-run groups. Pattern: {p}",
        "One flat class containing the hyphen would allow leading, trailing and repeated hyphens. "
        "Pattern: {p}",
    ],
    ("slug_chain", 2): [
        "Adding the underscore is a one-character change to the separator class. Pattern: {p}",
        "The surrounding structure already enforces the no-leading, no-trailing and no-repeat rules "
        "for whichever separator is used. Pattern: {p}",
        "Adding a whole second alternative for underscore-separated slugs would duplicate the grammar "
        "unnecessarily. Pattern: {p}",
    ],
    ("slug_chain", 3): [
        "A linter wants the complement: strings containing at least one disallowed character. "
        "Pattern: {p}",
        "Rather than negating the whole grammar, I assert that one out-of-class character exists "
        "somewhere, with anything either side. Pattern: {p}",
        "Negating the character class alone would only match strings made entirely of bad characters, "
        "not ones that merely contain some. Pattern: {p}",
    ],
    ("identifier_chain", 1): [
        "The head obeys a different rule from the tail: letter or underscore only, no digit. "
        "Pattern: {p}",
        "One class for the first character, a wider one with '*' for the rest. Pattern: {p}",
        "A single class for the whole string would accept a leading digit. Pattern: {p}",
    ],
    ("identifier_chain", 2): [
        "The rejected case I want is specifically a leading digit, so the head class becomes a digit "
        "and the tail stays. Pattern: {p}",
        "Inverting the head condition while leaving the rest of the construction intact. Pattern: {p}",
        "Negating the entire original pattern would match all sorts of strings that are not "
        "identifier-shaped at all. Pattern: {p}",
    ],
    ("identifier_chain", 3): [
        "The failure is a pure digit run being flagged, which is a number rather than a malformed "
        "name. Pattern: {p}",
        "Requiring at least one letter or underscore somewhere after the leading digit distinguishes "
        "the two. Pattern: {p}",
        "Forbidding digits in the tail would over-correct and reject genuinely malformed names that "
        "mix digits and letters. Pattern: {p}",
    ],
    ("hex_literal_chain", 1): [
        "The prefix is mandatory in either case, then one or more hex digits. Pattern: {p}",
        "Prefix spellings in a class, then '+' over the hex alphabet. Pattern: {p}",
        "Using '*' would accept a bare prefix with no digits. Pattern: {p}",
    ],
    ("hex_literal_chain", 2): [
        "Malformed means at least one letter beyond the hex range appears, so I keep the prefix and "
        "require one out-of-range character in the middle. Pattern: {p}",
        "Inverting the alphabet condition while keeping the literal prefix. Pattern: {p}",
        "Negating the hex class outright would also reject the digits that legitimately appear "
        "alongside the bad character. Pattern: {p}",
    ],
    ("hex_literal_chain", 3): [
        "The same construction transfers directly: only the prefix and the invalid alphabet change. "
        "Pattern: {p}",
        "Retargeting from hexadecimal to binary means a different prefix and a different out-of-range "
        "digit set. Pattern: {p}",
        "Reusing the hex out-of-range class would be wrong here, since for binary the invalid "
        "characters are digits rather than letters. Pattern: {p}",
    ],
    ("us_zip_chain", 1): [
        "Exactly five digits and nothing else, which one counted quantifier gives under fullmatch. "
        "Pattern: {p}",
        "A single counted run, no anchors needed. Pattern: {p}",
        "Using '+' would accept any length, which the fixed length rules out. Pattern: {p}",
    ],
    ("us_zip_chain", 2): [
        "The rejected lengths fall either side of five, so this is two branches: one to four digits, "
        "or six and above. Pattern: {p}",
        "No single quantifier expresses a gap, hence the alternation. Pattern: {p}",
        "A range from one to ten would include the valid length in the middle, defeating the point. "
        "Pattern: {p}",
    ],
    ("us_zip_chain", 3): [
        "Generalising means parameterising the length bound rather than adding branches. Pattern: {p}",
        "One bounded quantifier replaces the fixed count from the original rule. Pattern: {p}",
        "Enumerating each acceptable length as its own branch would work but misses the point of "
        "generalising. Pattern: {p}",
    ],
    ("md_header_chain", 1): [
        "Headers occupy their own lines, so MULTILINE makes the anchors line-relative. Pattern: {p} "
        "with replacement {r}",
        "The heading text is captured so the replacement can wrap it. Pattern: {p} with "
        "replacement {r}",
        "Without the required space a deeper header would match as this level. Pattern: {p} with "
        "replacement {r}",
    ],
    ("md_header_chain", 2): [
        "The repeat count on the marker becomes a bounded range instead of a single literal, which is "
        "the parameterisation being asked for. Pattern: {p} with replacement {r}",
        "Since the output tag is generic, the depth does not need capturing. Pattern: {p} with "
        "replacement {r}",
        "Writing six separate branches would generalise the behaviour but not the pattern, and would "
        "be far harder to extend. Pattern: {p} with replacement {r}",
    ],
    ("md_header_chain", 3): [
        "A second marker syntax is added as an alternation in the position the first occupied. "
        "Pattern: {p} with replacement {r}",
        "The required space and captured text carry over unchanged from the previous answer. "
        "Pattern: {p} with replacement {r}",
        "Running a second substitution afterwards would work but the task asks for one pattern "
        "handling both. Pattern: {p} with replacement {r}",
    ],
    ("quoted_chain", 1): [
        "A greedy middle would run from the first delimiter to the last one in the text, so a negated "
        "class is used instead. Pattern: {p}",
        "Literal delimiters with a negated class between them, group inside. Pattern: {p}",
        "Dot-star between quotes is the classic greedy mistake here. Pattern: {p}",
    ],
    ("quoted_chain", 2): [
        "Parameterising the delimiter means a class at both ends, and excluding all three characters "
        "from the content so the match still stops at the first closing one. Pattern: {p}",
        "One class replaces the single literal, on both sides and in the negation. Pattern: {p}",
        "Widening only the opening delimiter would let the match run past a closing one of a different "
        "type. Pattern: {p}",
    ],
    ("quoted_chain", 3): [
        "The reported capture spans from one delimiter to a different one, because nothing forces the "
        "pair to match and an empty run lets the scan start on the wrong delimiter. Pattern: {p}",
        "Excluding whitespace from the content rules that out here without needing a backreference, "
        "which would put the content in the second group. Pattern: {p}",
        "Requiring merely a non-empty run would not be enough, since the bad match had content in it "
        "already. Pattern: {p}",
    ],
    ("repeated_word_chain", 1): [
        "Detecting a repeat needs a backreference: the word is captured once and then required to "
        "occur again immediately. Pattern: {p}",
        "Capture, literal separator, backreference, with word boundaries either side. Pattern: {p}",
        "Two identical word classes would match any two adjacent words, not a repeat of the same one. "
        "Pattern: {p}",
    ],
    ("repeated_word_chain", 2): [
        "Generalising the gap is a one-token change, from a literal space to a whitespace class with "
        "'+'. Pattern: {p}",
        "The backreference and boundaries are untouched; only the separator widens. Pattern: {p}",
        "Enumerating a space, a tab and a newline as alternatives would work but the class already "
        "covers them. Pattern: {p}",
    ],
    ("repeated_word_chain", 3): [
        "The separator reverts to the single literal space. The new part is IGNORECASE, which also "
        "makes the backreference case-insensitive. Pattern: {p}",
        "Worth noting the flag affects the backreference too, so a capitalised word matches its "
        "lowercase repeat. Pattern: {p}",
        "Adding case alternatives to the word class would not help, since the repeat is matched by "
        "backreference rather than by the class. Pattern: {p}",
    ],
    ("file_extension_chain", 1): [
        "'Final dot' is the constraint that matters, since a plain dot-then-anything stops at the first "
        "dot of a multi-part name. Pattern: {p}",
        "Anchoring to the end and forbidding dots inside the group forces the match onto the last one. "
        "Pattern: {p}",
        "Without the end anchor the match would settle on the first dot. Pattern: {p}",
    ],
    ("file_extension_chain", 2): [
        "For a multi-part extension the group must start at the first dot of the final segment rather "
        "than the last dot overall. Pattern: {p}",
        "Consuming a leading run that excludes both dots and slashes positions the match on that "
        "segment, and the group takes everything after its first dot. Pattern: {p}",
        "Relaxing the group to allow dots is not enough on its own; the match also has to be anchored "
        "to the right dot to begin with. Pattern: {p}",
    ],
    ("file_extension_chain", 3): [
        "The complement is a name containing no dot at all. Pattern: {p}",
        "Anchoring both ends and excluding the dot from the class expresses that directly, with the "
        "whole name captured. Pattern: {p}",
        "Negating the previous pattern is not expressible directly; describing the complementary shape "
        "is. Pattern: {p}",
    ],
}
