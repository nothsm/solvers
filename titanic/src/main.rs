#![allow(warnings)]

use std::fs::File;
use std::io::Read;

/*
 * TODO: Is there a simpler way to write these?
 */
const DATA_DIR: &str = "data/";
const TRAIN_FILE: &str = "train.csv";
const TEST_FILE: &str = "test.csv";

const N_VARIABLES: usize = 12;

/*
 * TODO: These 2 functions do not feel like The Right Thing.
 */
fn train_file() -> String {
    String::from(DATA_DIR) + TRAIN_FILE
}

fn test_file() -> String {
    String::from(DATA_DIR) + TEST_FILE
}

/*
 * TODO: This function is really unsafe right now.
 */
fn slurp(file: &str) -> std::io::Result<String> {
    /*
     * Pre: TODO
     */
    let mut f = File::open(train_file())?;
    let mut buf = String::new();
    f.read_to_string(&mut buf)?;
    /*
     * Post: TODO
     */
    Ok(buf)
}

/*
 * TODO: Change this repr
 */
#[repr(C)]
#[derive(Debug)]
struct Variable {
    id: u32,
    p_class: u32,
    name: String,
    sex: String,
    age: f32,
    sibsp: u32,
    p_arch: u32,
    ticket: String,
    fare: f32,
    cabin: Option<String>,
    embarked: String,
}

#[repr(C)]
struct Objective {
    id: u32,
    survived: bool,
}

/*
 * TODO: This could be more generic, and more efficient.
 */
fn fields(s: &str) -> Vec<&str> {
    /*
     * Pre: TODO
     */
    s.split(',').collect()
    /*
     * Post: TODO
     */
}

fn make_variable(
    id: u32,
    survived: bool,
    p_class: u32,
    name: String,
    sex: String,
    age: f32,
    sibsp: u32,
    p_arch: u32,
    ticket: String,
    fare: f32,
    cabin: Option<String>,
    embarked: String,
) -> Variable {
    Variable {
        id: id,
        p_class: p_class,
        name: name,
        sex: sex,
        age: age,
        sibsp: sibsp,
        p_arch: p_arch,
        ticket: ticket,
        fare: fare,
        cabin: cabin,
        embarked: embarked,
    }
}

/*
 * TODO: Modify to return a generic iterator I can .collect().
 */
fn record_split(s: &str) -> Vec<String> {
    /*
     * Pre: TODO
     */
    let cs = s.as_bytes();
    let n = cs.len();

    let mut start = 0;
    let mut acc = vec![];
    /*
     * Inv: TODO
     *
     * TODO: Refactor this loop.
     */
    while start < n {
        if cs[start] == b'\"' {
            start += 1; /* Skip first quotation mark */

            let mut end = start;
            let delim = b'\"';
            /*
             * Inv: TODO
             */
            while end < n && cs[end] != delim {
                end += 1;
            }
            acc.push(String::from_utf8_lossy(&cs[start..end]).into_owned());
            start = end + 2; /* Skip last quotation mark, and comma */
        } else {
            let mut end = start;
            let delim = b',';
            /*
             * Inv: TODO
             */
            while end < n && cs[end] != delim {
                end += 1;
            }
            /*
             * TODO: Fix this.
             */
            acc.push(String::from_utf8_lossy(&cs[start..end]).into_owned());
            start = end + 1; /* Skip comma */
        }
    }
    /*
     * Post: TODO
     */
    acc
}

fn variable_parse(s: &str) -> Result<Variable, &'static str> {
    /*
     * Pre: TODO
     */
    let fields = &record_split(s)[..];
    debug_assert!(fields.len() == N_VARIABLES);

    // let [id, survived, p_class]
    todo!()
    /*
     * Post: TODO
     */
}

/*
 * TODO: Update this to use a (eventually custom) hashmap instead.
 */
fn csv_parse(s: &str) -> Vec<Variable> {
    /*
     * Pre: TODO
     */

    unimplemented!();
    // let ls = s.lines().skip(1);
    // ls.map(variable_parse).collect()
    /*
     * Post: TODO
     */
}

fn main() -> std::io::Result<()> {
    let contents = slurp(&train_file())?;
    println!("{}", &contents[..1000]);

    // let dataset = csv_parse(&contents);
    // println!("{:#?}", &dataset[..10]);

    Ok(())
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use expect_test::{Expect, expect};

    use super::*;

    fn str_check<T: ToString>(actual: T, expect: Expect) {
        let actual = actual.to_string();
        expect.assert_eq(&actual);
    }

    fn dbg_check<T: Debug>(actual: T, expect: Expect) {
        let actual = format!("{:?}", actual);
        expect.assert_eq(&actual);
    }

    fn ppr_check<T: Debug>(actual: T, expect: Expect) {
        let actual = format!("{:#?}", actual);
        expect.assert_eq(&actual);
    }

    #[test]
    fn record_split_test() {
        let s = r#"1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S"#;
        ppr_check(
            record_split(s),
            expect![[r#"
            [
                "1",
                "0",
                "3",
                "Braund, Mr. Owen Harris",
                "male",
                "22",
                "1",
                "0",
                "A/5 21171",
                "7.25",
                "",
                "S",
            ]"#]],
        );
    }

    #[test]
    fn variable_parse_test() {
        let s = r#"1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S"#;
        dbg_check(variable_parse(s).unwrap(), expect![]);
    }
}

/*
 * TODO:
 * - [ ] Write custom kaggle CLI tool for fetching data.
 * - [ ] How is String implemented?
 * - [ ] How is String::from implemented?
 * - [ ] How is String::add implemented?
 * - [ ] How is File implemented?
 * - [ ] How is File::create implemented?
 * - [ ] How is Vec implemented?
 * - [ ] Write custom String::lines() function.
 */
