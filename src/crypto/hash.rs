use sha2::{Sha256, Digest};

pub fn sha256(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn md5(_input: &str) -> String {
    String::new() // MD5 not needed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        let input = "Hello, world!".to_string();
        let hash = sha256(&input);
        println!("{}", hash);
        assert_eq!(hash, "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3");
    }
}
