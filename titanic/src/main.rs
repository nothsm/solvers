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
const N_TC: usize = 3; /* TODO: I should be able to compute this dynamically. */
const N_SEX: usize = 2;
const N_PORT: usize = 3;

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
 * TODO: Move associated functions to methods.
 */
#[repr(C)]
#[derive(Debug)]
enum TicketClass {
    First = 0,
    Second = 1,
    Third = 2,
}

/*
 * TODO: There must be a shorter way to write this.
 */
fn tc_from_int(x: i32) -> Result<TicketClass, &'static str> {
    /*
     * Pre: TODO
     */
    match x {
        0 => Ok(TicketClass::First),
        1 => Ok(TicketClass::Second),
        2 => Ok(TicketClass::Third),
        _ => Err("int_to_tc: given int must be 0, 1, or 2"),
    }
    /*
     * Post: TODO
     */
}

fn int_from_tc(t: TicketClass) -> i32 {
    use TicketClass::{First, Second, Third};
    /*
     * Pre: TODO
     */
    match t {
        First => 0,
        Second => 1,
        Third => 2,
    }
    /*
     * Post: TODO
     */
}

/*
 * TODO: The error message in this function exposes implementation details.
 */
fn tc_parse(s: &str) -> Result<TicketClass, &'static str> {
    /*
     * Pre: TODO
     */
    if let Ok(n) = s.parse::<i32>() {
        tc_from_int(n - 1)
    } else {
        Err(r#"tc_parse: given string must be "1", "2", or "3""#)
    }
    /*
     * Post: TODO
     */
}

fn tc_one_hot(t: TicketClass) -> [f32; N_TC] {
    /*
     * Pre: TODO
     */
    let mut xs = [0.; N_TC];
    xs[int_from_tc(t) as usize] = 1.;
    xs
    /*
     * Post: TODO
     */
}

#[repr(C)]
#[derive(Debug)]
enum Sex {
    Male = 0,
    Female = 1,
}

fn sex_from_int(x: i32) -> Result<Sex, &'static str> {
    /*
     * Pre: TODO
     */
    use Sex::{Female, Male};

    match x {
        0 => Ok(Male),
        1 => Ok(Female),
        _ => Err("sex_from_int: unexpected int"),
    }
    /*
     * Post: TODO
     */
}

fn int_from_sex(s: Sex) -> i32 {
    /*
     * Pre: TODO
     */
    use Sex::{Female, Male};

    match s {
        Male => 0,
        Female => 1,
    }
    /*
     * Post: TODO
     */
}

fn sex_parse(s: &str) -> Result<Sex, &'static str> {
    /*
     * Pre: TODO
     */
    use Sex::{Female, Male};

    match s {
        "male" => Ok(Male),
        "female" => Ok(Female),
        _ => Err("sex_parse: unexpected string"),
    }
    /*
     * Post: TODO
     */
}

fn sex_one_hot(s: Sex) -> [f32; N_SEX] {
    /*
     * Pre: TODO
     */
    let mut xs = [0.; N_SEX];
    xs[int_from_sex(s) as usize] = 1.;
    xs
    /*
     * Post: TODO
     */
}

#[repr(C)]
#[derive(Debug)]
enum Port {
    C = 0, /* Cherbourg */
    Q = 1, /* Queenstown */
    S = 2, /* Southampton */
}

fn port_from_int(x: i32) -> Result<Port, &'static str> {
    /*
     * Pre: TODO
     */
    use Port::{C, Q, S};

    match x {
        0 => Ok(C),
        1 => Ok(Q),
        2 => Ok(S),
        _ => Err("port_from_int: unexpected int"),
    }
    /*
     * Post: TODO
     */
}

fn int_from_port(p: Port) -> i32 {
    /*
     * Pre: TODO
     */
    use Port::{C, Q, S};

    match p {
        C => 0,
        Q => 1,
        S => 2,
    }
    /*
     * Post: TODO
     */
}

fn port_parse(s: &str) -> Result<Port, &'static str> {
    /*
     * Pre: TODO
     */
    use Port::{C, Q, S};

    match s {
        "C" => Ok(C),
        "Q" => Ok(Q),
        "S" => Ok(S),
        _ => Err("port_parse: unexpected string"),
    }
    /*
     * Post: TODO
     */
}

fn port_one_hot(p: Port) -> [f32; N_PORT] {
    /*
     * Pre: TODO
     */
    let mut xs = [0.; N_PORT];
    xs[int_from_port(p) as usize] = 1.;
    xs
    /*
     * Post: TODO
     */
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
    tc: TicketClass,
    name: String,
    sex: Sex,
    age: Option<f32>,
    sibsp: u32,     /* Sibling + spouse count */
    parch: u32,     /* Parent + child count */
    tickno: String, /* Ticket number */
    fare: f32,
    cabno: Option<String>, /* Cabin number */
    port: Port,            /* Port of Embarkation */
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
    tc: TicketClass,
    name: &str,
    sex: Sex,
    age: Option<f32>,
    sibsp: u32,
    parch: u32,
    tickno: &str,
    fare: f32,
    cabno: Option<&str>,
    port: Port,
) -> Variable {
    let name = name.to_string();
    let tickno = tickno.to_string();
    let cabno = cabno.map(|s| s.to_string());
    Variable {
        id,
        survived,
        tc,
        name,
        sex,
        age,
        sibsp,
        parch,
        tickno,
        fare,
        cabno,
        port,
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
        tc,
        name,
        sex,
        age,
        sibsp,
        parch,
        ticket,
        fare,
        cabno,
        port,
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
        tc_parse(tc).expect("variable_parse: failed to parse tc"),
        name,
        sex_parse(sex).expect("variable_parse: failed to parse sex"),
        if let Ok(a) = age.parse() {
            Some(a)
        } else {
            None
        },
        sibsp
            .parse()
            .expect("variable_parse: failed to parse sibsp"),
        parch
            .parse()
            .expect("variable_parse: failed to parse parch"),
        ticket,
        fare.parse().expect("variable_parse: failed to parse fare"),
        if cabno.len() > 0 { Some(cabno) } else { None },
        port_parse(port).expect("variable_parse: failed to parse port"),
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
    println!("{:#?}", &dataset[..10]);

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
                    tc: Third,
                    name: "Braund, Mr. Owen Harris",
                    sex: Male,
                    age: Some(
                        22.0,
                    ),
                    sibsp: 1,
                    parch: 0,
                    tickno: "A/5 21171",
                    fare: 7.25,
                    cabno: None,
                    port: S,
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
                    tc: Third,
                    name: "Moran, Mr. James",
                    sex: Male,
                    age: None,
                    sibsp: 0,
                    parch: 0,
                    tickno: "330877",
                    fare: 8.4583,
                    cabno: None,
                    port: Q,
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
                        tc: Third,
                        name: "Braund, Mr. Owen Harris",
                        sex: Male,
                        age: Some(
                            22.0,
                        ),
                        sibsp: 1,
                        parch: 0,
                        tickno: "A/5 21171",
                        fare: 7.25,
                        cabno: None,
                        port: "S",
                    },
                    Variable {
                        id: 2,
                        survived: true,
                        tc: First,
                        name: "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                        sex: Female,
                        age: Some(
                            38.0,
                        ),
                        sibsp: 1,
                        parch: 0,
                        tickno: "PC 17599",
                        fare: 71.2833,
                        cabno: Some(
                            "C85",
                        ),
                        port: "C",
                    },
                    Variable {
                        id: 3,
                        survived: true,
                        tc: Third,
                        name: "Heikkinen, Miss. Laina",
                        sex: Female,
                        age: Some(
                            26.0,
                        ),
                        sibsp: 0,
                        parch: 0,
                        tickno: "STON/O2. 3101282",
                        fare: 7.925,
                        cabno: None,
                        port: "S",
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
                        tc: First,
                        name: "Graham, Miss. Margaret Edith",
                        sex: Female,
                        age: Some(
                            19.0,
                        ),
                        sibsp: 0,
                        parch: 0,
                        tickno: "112053",
                        fare: 30.0,
                        cabno: Some(
                            "B42",
                        ),
                        port: "S",
                    },
                    Variable {
                        id: 889,
                        survived: false,
                        tc: Third,
                        name: "Johnston, Miss. Catherine Helen \"\"Carrie\"\"",
                        sex: Female,
                        age: None,
                        sibsp: 1,
                        parch: 2,
                        tickno: "W./C. 6607",
                        fare: 23.45,
                        cabno: None,
                        port: "S",
                    },
                    Variable {
                        id: 890,
                        survived: true,
                        tc: First,
                        name: "Behr, Mr. Karl Howell",
                        sex: Male,
                        age: Some(
                            26.0,
                        ),
                        sibsp: 0,
                        parch: 0,
                        tickno: "111369",
                        fare: 30.0,
                        cabno: Some(
                            "C148",
                        ),
                        port: "C",
                    },
                ]"#]],
        );
    }

    #[test]
    fn tc_from_int_test() {
        let x = 0;
        dbg_check(tc_from_int(x), expect!["Ok(First)"]);

        let x = 1;
        dbg_check(tc_from_int(x), expect!["Ok(Second)"]);

        let x = 2;
        dbg_check(tc_from_int(x), expect!["Ok(Third)"]);

        let x = 3;
        dbg_check(
            tc_from_int(x),
            expect![[r#"Err("int_to_tc: given int must be 0, 1, or 2")"#]],
        );
    }

    #[test]
    fn int_from_tc_test() {
        use TicketClass::{First, Second, Third};

        str_check(int_from_tc(First), expect!["0"]);
        str_check(int_from_tc(Second), expect!["1"]);
        str_check(int_from_tc(Third), expect!["2"]);
    }

    #[test]
    fn tc_parse_test() {
        let s = "0";
        dbg_check(
            tc_parse(s),
            expect![[r#"Err("int_to_tc: given int must be 0, 1, or 2")"#]],
        );

        let s = "1";
        dbg_check(tc_parse(s), expect!["Ok(First)"]);

        let s = "2";
        dbg_check(tc_parse(s), expect!["Ok(Second)"]);

        let s = "3";
        dbg_check(tc_parse(s), expect!["Ok(Third)"]);

        let s = "4";
        dbg_check(
            tc_parse(s),
            expect![[r#"Err("int_to_tc: given int must be 0, 1, or 2")"#]],
        );
    }

    #[test]
    fn tc_one_hot_test() {
        use TicketClass::{First, Second, Third};

        dbg_check(tc_one_hot(First), expect!["[1.0, 0.0, 0.0]"]);
        dbg_check(tc_one_hot(Second), expect!["[0.0, 1.0, 0.0]"]);
        dbg_check(tc_one_hot(Third), expect!["[0.0, 0.0, 1.0]"]);
    }

    #[test]
    fn sex_from_int_test() {
        let x = 0;
        dbg_check(sex_from_int(x), expect!["Ok(Male)"]);

        let x = 1;
        dbg_check(sex_from_int(x), expect!["Ok(Female)"]);

        let x = 2;
        dbg_check(
            sex_from_int(x),
            expect![[r#"Err("sex_from_int: unexpected int")"#]],
        );

        let x = 3;
        dbg_check(
            sex_from_int(x),
            expect![[r#"Err("sex_from_int: unexpected int")"#]],
        );
    }

    #[test]
    fn int_from_sex_test() {
        use Sex::{Female, Male};

        str_check(int_from_sex(Male), expect!["0"]);
        str_check(int_from_sex(Female), expect!["1"]);
    }

    #[test]
    fn sex_parse_test() {
        let s = "0";
        dbg_check(
            sex_parse(s),
            expect![[r#"Err("sex_parse: unexpected string")"#]],
        );

        let s = "1";
        dbg_check(
            sex_parse(s),
            expect![[r#"Err("sex_parse: unexpected string")"#]],
        );

        let s = "male";
        dbg_check(sex_parse(s), expect!["Ok(Male)"]);

        let s = "female";
        dbg_check(sex_parse(s), expect!["Ok(Female)"]);

        let s = "4";
        dbg_check(
            sex_parse(s),
            expect![[r#"Err("sex_parse: unexpected string")"#]],
        );
    }

    #[test]
    fn sex_one_hot_test() {
        use Sex::{Female, Male};

        dbg_check(sex_one_hot(Male), expect!["[1.0, 0.0]"]);
        dbg_check(sex_one_hot(Female), expect!["[0.0, 1.0]"]);
    }

    #[test]
    fn port_from_int_test() {
        let x = 0;
        dbg_check(port_from_int(x), expect!["Ok(C)"]);

        let x = 1;
        dbg_check(port_from_int(x), expect!["Ok(Q)"]);

        let x = 2;
        dbg_check(port_from_int(x), expect!["Ok(S)"]);

        let x = 3;
        dbg_check(
            port_from_int(x),
            expect![[r#"Err("port_from_int: unexpected int")"#]],
        );
    }

    #[test]
    fn int_from_port_test() {
        use Port::{C, Q, S};

        str_check(int_from_port(C), expect!["0"]);
        str_check(int_from_port(Q), expect!["1"]);
        str_check(int_from_port(S), expect!["2"]);
    }

    #[test]
    fn port_parse_test() {
        let s = "0";
        dbg_check(
            port_parse(s),
            expect![[r#"Err("port_parse: unexpected string")"#]],
        );

        let s = "1";
        dbg_check(
            port_parse(s),
            expect![[r#"Err("port_parse: unexpected string")"#]],
        );

        let s = "C";
        dbg_check(port_parse(s), expect!["Ok(C)"]);

        let s = "Q";
        dbg_check(port_parse(s), expect!["Ok(Q)"]);

        let s = "S";
        dbg_check(port_parse(s), expect!["Ok(S)"]);
    }

    #[test]
    fn port_one_hot_test() {
        use Port::{C, Q, S};

        dbg_check(port_one_hot(C), expect!["[1.0, 0.0, 0.0]"]);
        dbg_check(port_one_hot(Q), expect!["[0.0, 1.0, 0.0]"]);
        dbg_check(port_one_hot(S), expect!["[0.0, 0.0, 1.0]"]);
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
 * - [ ] Writing enum's for each categorical variable is horrifically slow.
 *       I should be able to generate these with macros.
 * - [ ] What is the runtime representation of enum's?
 */
