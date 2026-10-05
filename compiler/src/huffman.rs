//! Canonical Huffman with length limit 8 (single-LUT decode).
//! Codes are assigned MSB-first canonical; the bitstream writes them
//! MSB-first sequentially, so an LSB-first peek of L bits forms the
//! bit-reversed code index — the decoder builds its table accordingly.

#[derive(Clone)]
pub struct HuffTable {
    pub lens: [u8; 256], // 0 = symbol unused
}

pub fn build(freqs: &[u32; 256]) -> HuffTable {
    let mut f = *freqs;
    loop {
        let lens = build_unlimited(&f);
        let maxlen = lens.iter().copied().max().unwrap_or(0);
        if maxlen <= 8 {
            return HuffTable { lens };
        }
        // scale down rare symbols' frequencies and retry
        for x in f.iter_mut() {
            if *x > 0 { *x = (*x + 1) / 2; }
        }
    }
}

struct Node {
    freq: u64,
    sym: i32, // -1 for internal
    left: usize,
    right: usize,
}

fn build_unlimited(freqs: &[u32; 256]) -> [u8; 256] {
    let mut lens = [0u8; 256];
    let mut nodes: Vec<Node> = Vec::new();
    let mut live: Vec<usize> = Vec::new();
    for (sym, &fr) in freqs.iter().enumerate() {
        if fr > 0 {
            nodes.push(Node { freq: fr as u64, sym: sym as i32, left: usize::MAX, right: usize::MAX });
            live.push(nodes.len() - 1);
        }
    }
    if live.is_empty() {
        return lens; // nothing to encode
    }
    if live.len() == 1 {
        lens[nodes[live[0]].sym as usize] = 1;
        return lens;
    }
    // simple O(n^2) Huffman (n <= 62 opcode symbols — fine)
    while live.len() > 1 {
        // two smallest
        let mut a = 0usize;
        let mut b = 1usize;
        if nodes[live[b]].freq < nodes[live[a]].freq { std::mem::swap(&mut a, &mut b); }
        for i in 2..live.len() {
            if nodes[live[i]].freq < nodes[live[a]].freq {
                b = a; a = i;
            } else if nodes[live[i]].freq < nodes[live[b]].freq {
                b = i;
            }
        }
        let ia = live[a];
        let ib = live[b];
        let freq = nodes[ia].freq + nodes[ib].freq;
        nodes.push(Node { freq, sym: -1, left: ia, right: ib });
        let new_idx = nodes.len() - 1;
        // remove a,b from live (a<b positions in vec, remove larger first)
        let (pa, pb) = if a < b { (a, b) } else { (b, a) };
        live.remove(pb);
        live.remove(pa);
        live.push(new_idx);
    }
    // depth walk
    fn walk(nodes: &[Node], idx: usize, depth: u32, lens: &mut [u8; 256]) {
        let n = &nodes[idx];
        if n.sym >= 0 {
            let l = depth.max(1) as u8;
            lens[n.sym as usize] = l;
        } else {
            walk(nodes, n.left, depth + 1, lens);
            walk(nodes, n.right, depth + 1, lens);
        }
    }
    walk(&nodes, live[0], 0, &mut lens);
    lens
}

/// Canonical codes: sorted by (len, symbol). Returns (code, len) per symbol.
/// Standard algorithm: next_code = (prev_code + 1) << (len - prev_len).
pub fn canonical_codes(table: &HuffTable) -> Vec<(u16, u8)> {
    let mut out = vec![(0u16, 0u8); 256];
    let mut syms: Vec<u8> = (0..256u16).map(|s| s as u8)
        .filter(|&s| table.lens[s as usize] > 0).collect();
    syms.sort_by_key(|&s| (table.lens[s as usize], s));
    let mut code: u32 = 0;
    let mut prev_len: u8 = 0;
    for &s in &syms {
        let l = table.lens[s as usize];
        if prev_len == 0 {
            prev_len = l;
        } else {
            code = (code + 1) << (l - prev_len);
            prev_len = l;
        }
        out[s as usize] = (code as u16, l);
    }
    out
}

/// Reverse the low `len` bits of `code` (for LSB-first decode table indexing).
pub fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut r = 0u16;
    for i in 0..len {
        r |= ((code >> i) & 1) << (len - 1 - i);
    }
    r
}

/// Stream decode for a canonical-Huffman bitstream (dictionary synthesis):
/// reconstruct codes from (sym -> len) exactly like `canonical_codes`, then
/// MSB-first peek-decode `total` bytes. Used by the disasm for packed text
/// payloads; the Zig loader runs the identical algorithm.
pub fn decode_stream(bytes: &[u8], total: usize, lens: &[u8; 256]) -> Result<Vec<u8>, String> {
    // canonical code assignment (mirror of canonical_codes)
    let mut codes = vec![(0u16, 0u8); 256];
    let mut syms: Vec<u8> = (0..256u16).map(|s| s as u8)
        .filter(|&s| lens[s as usize] > 0).collect();
    syms.sort_by_key(|&s| (lens[s as usize], s));
    let mut code: u32 = 0;
    let mut prev_len: u8 = 0;
    for &s in &syms {
        let l = lens[s as usize];
        if prev_len == 0 { prev_len = l; } else { code = (code + 1) << (l - prev_len); prev_len = l; }
        if l > 8 { return Err(format!("huffman code length {} > 8 unsupported", l)); }
        codes[s as usize] = (code as u16, l);
    }
    // decode LUT over 8-bit peek windows
    let mut lut = vec![0u16; 256]; // sym << 8 | len
    for s in 0..256u32 {
        let l = lens[s as usize];
        if l == 0 || l > 8 { continue; }
        let mut base: u32 = 0;
        for b in 0..l {
            base |= (((codes[s as usize].0 >> (l - 1 - b)) & 1) as u32) << b;
        }
        let span = 1u32 << (8 - l);
        for j in 0..span {
            lut[(base + (j << l)) as usize] = ((s as u16) << 8) | l as u16;
        }
    }
    let mut out = Vec::with_capacity(total);
    let mut bitpos: u32 = 0;
    while out.len() < total {
        let mut peek: u32 = 0;
        for b in 0..8u32 {
            let byte = (bitpos + b) / 8;
            if (byte as usize) < bytes.len() {
                peek |= (((bytes[byte as usize] >> (bitpos + b) % 8) & 1) as u32) << b;
            }
        }
        let entry = lut[peek as usize];
        let l = (entry & 0xFF) as u32;
        if l == 0 { return Err(format!("invalid huffman stream at bit {}", bitpos)); }
        bitpos += l;
        out.push((entry >> 8) as u8);
    }
    Ok(out)
}
