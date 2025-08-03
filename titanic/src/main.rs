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
 * TODO: Change this repr.
 * TODO: I should use Struct-of-Arrays rather than an Array-of-Structs.
 */
#[repr(C)]
#[derive(Debug)]
struct Variable {
    id: i128,
    survived: bool,
    ticket_class: i128,
    name: String,
    sex: String,
    age: Option<f64>,
    sibling_spouse: i128,
    parent_child: i128,
    ticket_no: String,
    fare: f64,
    cabin_no: Option<String>,
    embark_port: String,
}

#[repr(C)]
struct Objective {
    id: u32,
    survived: bool,
}

/*
 * TODO: This procedure performs a LOT of unnecessary cloning. How can I make
 *       this more efficient?
 */
fn make_variable(
    id: i128,
    survived: bool,
    ticket_class: i128,
    name: &str,
    sex: &str,
    age: Option<f64>,
    sibling_spouse: i128,
    parent_child: i128,
    ticket_no: &str,
    fare: f64,
    cabin_no: Option<&str>,
    embark_port: &str,
) -> Variable {
    let name = name.to_string();
    let sex = sex.to_string();
    let ticket_no = ticket_no.to_string();
    let cabin_no = cabin_no.map(|s| s.to_string());
    let embark_port = embark_port.to_string();
    Variable {
        id,
        survived,
        ticket_class,
        name,
        sex,
        age,
        sibling_spouse,
        parent_child,
        ticket_no,
        fare,
        cabin_no,
        embark_port,
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
             *
             * TODO: This loop is super unsafe.
             */
            while end < n && !(cs[end] == delim && cs[end + 1] == b',') {
                end += 1;
            }
            acc.push(String::from_utf8_lossy(&cs[start..end]).into_owned());
            start = end + 2; /* Skip last (1) quotation mark, and (2) comma */
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
            start = end + 1; /* Skip (1) comma */
        }
    }
    /*
     * TODO: Why is this correct? (special case where last field is empty)
     */
    if cs[n - 1] == b',' {
        acc.push("".to_string());
    }
    /*
     * Post: TODO
     */
    acc
}

/*
 * TODO: This function is unsafe. It should properly handle errors.
 */
fn variable_parse(s: &str) -> Result<Variable, &'static str> {
    /*
     * Pre: TODO
     */
    let fields = &record_split(s)[..];
    debug_assert!(fields.len() == N_VARIABLES);

    let [
        id,
        survived,
        ticket_class,
        name,
        sex,
        age,
        sibling_spouse,
        parent_child,
        ticket,
        fare,
        cabin_no,
        embark_port,
    ] = fields
    else {
        todo!()
    };

    Ok(make_variable(
        id.parse().expect("variable_parse: failed to parse id"),
        survived
            .parse::<u32>()
            .expect("variable_parse: failed to parse survived")
            != 0,
        ticket_class
            .parse()
            .expect("variable_parse: failed to parse ticket_class"),
        name,
        sex,
        if let Ok(a) = age.parse() {
            Some(a)
        } else {
            None
        },
        sibling_spouse
            .parse()
            .expect("variable_parse: failed to parse sibling_spouse"),
        parent_child
            .parse()
            .expect("variable_parse: failed to parse parent_child"),
        ticket,
        fare.parse().expect("variable_parse: failed to parse fare"),
        if cabin_no.len() > 0 {
            Some(cabin_no)
        } else {
            None
        },
        embark_port,
    ))
    /*
     * Post: TODO
     */
}

/*
 * TODO: Update this to use a (eventually custom) hashmap instead.
 */
fn csv_parse(s: &str) -> Result<Vec<Variable>, &'static str> {
    /*
     * Pre: TODO
     */
    let mut acc = vec![];
    /*
     * Inv: TODO
     */
    for line in s.lines().skip(1) {
        acc.push(variable_parse(line)?);
    }
    Ok(acc)
    /*
     * Post: TODO
     */
}

fn main() -> std::io::Result<()> {
    let contents = slurp(&train_file())?;
    println!("{}", &contents[..1000]);

    let dataset = csv_parse(&contents).expect("main: csv parse error");
    println!("{:#?}", &dataset[..]);

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
    /*
     * TODO: Add a test case for missing last field.
     */
    fn record_split_test() {
        let s = r#"1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S"#;
        let record = record_split(s);
        ppr_check(
            record,
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
        let var = variable_parse(s).unwrap();
        ppr_check(
            var,
            expect![[r#"
                Variable {
                    id: 1,
                    survived: false,
                    ticket_class: 3,
                    name: "Braund, Mr. Owen Harris",
                    sex: "male",
                    age: Some(
                        22.0,
                    ),
                    sibling_spouse: 1,
                    parent_child: 0,
                    ticket_no: "A/5 21171",
                    fare: 7.25,
                    cabin_no: None,
                    embark_port: "S",
                }"#]],
        );

        let s = r#"6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q"#;
        let var = variable_parse(s).unwrap();
        ppr_check(
            var,
            expect![[r#"
                Variable {
                    id: 6,
                    survived: false,
                    ticket_class: 3,
                    name: "Moran, Mr. James",
                    sex: "male",
                    age: None,
                    sibling_spouse: 0,
                    parent_child: 0,
                    ticket_no: "330877",
                    fare: 8.4583,
                    cabin_no: None,
                    embark_port: "Q",
                }"#]],
        )
    }

    #[test]
    fn csv_parse_test() {
        let s = slurp(&train_file()).unwrap();

        let csv = csv_parse(&s).unwrap();
        let N = csv.len();

        let n = 3;

        let first_n = &csv[..n];
        ppr_check(
            first_n,
            expect![[r#"
                [
                    Variable {
                        id: 1,
                        survived: false,
                        ticket_class: 3,
                        name: "Braund, Mr. Owen Harris",
                        sex: "male",
                        age: Some(
                            22.0,
                        ),
                        sibling_spouse: 1,
                        parent_child: 0,
                        ticket_no: "A/5 21171",
                        fare: 7.25,
                        cabin_no: None,
                        embark_port: "S",
                    },
                    Variable {
                        id: 2,
                        survived: true,
                        ticket_class: 1,
                        name: "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                        sex: "female",
                        age: Some(
                            38.0,
                        ),
                        sibling_spouse: 1,
                        parent_child: 0,
                        ticket_no: "PC 17599",
                        fare: 71.2833,
                        cabin_no: Some(
                            "C85",
                        ),
                        embark_port: "C",
                    },
                    Variable {
                        id: 3,
                        survived: true,
                        ticket_class: 3,
                        name: "Heikkinen, Miss. Laina",
                        sex: "female",
                        age: Some(
                            26.0,
                        ),
                        sibling_spouse: 0,
                        parent_child: 0,
                        ticket_no: "STON/O2. 3101282",
                        fare: 7.925,
                        cabin_no: None,
                        embark_port: "S",
                    },
                ]"#]],
        );

        let last_n = &csv[((N - n) - 1)..(N - 1)];
        ppr_check(
            last_n,
            expect![[r#"
                [
                    Variable {
                        id: 888,
                        survived: true,
                        ticket_class: 1,
                        name: "Graham, Miss. Margaret Edith",
                        sex: "female",
                        age: Some(
                            19.0,
                        ),
                        sibling_spouse: 0,
                        parent_child: 0,
                        ticket_no: "112053",
                        fare: 30.0,
                        cabin_no: Some(
                            "B42",
                        ),
                        embark_port: "S",
                    },
                    Variable {
                        id: 889,
                        survived: false,
                        ticket_class: 3,
                        name: "Johnston, Miss. Catherine Helen \"\"Carrie\"\"",
                        sex: "female",
                        age: None,
                        sibling_spouse: 1,
                        parent_child: 2,
                        ticket_no: "W./C. 6607",
                        fare: 23.45,
                        cabin_no: None,
                        embark_port: "S",
                    },
                    Variable {
                        id: 890,
                        survived: true,
                        ticket_class: 1,
                        name: "Behr, Mr. Karl Howell",
                        sex: "male",
                        age: Some(
                            26.0,
                        ),
                        sibling_spouse: 0,
                        parent_child: 0,
                        ticket_no: "111369",
                        fare: 30.0,
                        cabin_no: Some(
                            "C148",
                        ),
                        embark_port: "C",
                    },
                ]"#]],
        );
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
 * - [ ] Fix project structure in emacs.
 */
