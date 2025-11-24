static MAGIC_SQUARE: [u32; 9] = [2, 9, 4, 7, 5, 3, 6, 1, 8];

static TWO_POWERS: [u32; 10] = two_powers_new();

/* TODO: fix this naming... use "combinations" */
fn any_n_sum_to_k(n: u32, k: i32, xs: &[u32]) -> bool {
    fn any_n_sum_to_k_search(i: usize, n: u32, k: i32, xs: &[u32]) -> bool {
        if n == 0 {
            k == 0
        } else if k < 0 {
            false
        } else if xs.len() - i == 0 {
            false
        } else if any_n_sum_to_k_search(i + 1, n - 1, k - (xs[i] as i32), xs) {
            true
        } else if any_n_sum_to_k_search(i + 1, n, k, xs) {
            true
        } else {
            false
        }
    }
    any_n_sum_to_k_search(0, n, k, xs)
}

/* TODO: make this return a String */
fn state_display(state: &(Vec<u32>, Vec<u32>, usize) ) {
    let (x_locs, o_locs, _) = state;
    println!();
    for (i, loc) in MAGIC_SQUARE.iter().enumerate() {
        if x_locs.contains(&loc) {
            print!(" X");
        } else if o_locs.contains(&loc) {
            print!(" O")
        } else {
            print!(" -")
        }
        if i == 5 {
            print!("  {:.3}", 0.0); /* TODO: use value() here */
        }
        if i.rem_euclid(3) == 2 {
            println!();
        }
    }
}

const fn two_powers_new() -> [u32; 10] {
    let mut powers = [0; 10];
    let mut i = 0;
    while i < 10 {
        powers[i] = 1 << i;
        i += 1;
    }
    powers
}


fn state_index(x_locs: &[u32], o_locs: &[u32]) -> u32 {
    let x_sum: u32  = x_locs
        .into_iter()
        .map(|loc| TWO_POWERS[*loc as usize])
        .sum();
    let o_sum: u32 = o_locs
        .into_iter()
        .map(|loc| TWO_POWERS[*loc as usize])
        .sum();
    x_sum + 512 * o_sum
}

/* TODO */
const VALUE_TABLE: [u32; 0] = [];
const INITIAL_STATE: () = ();

fn main() {
    let init_x_locs = vec![1, 2, 3];
    let init_o_locs = vec![4, 5, 6];
    let init_index = state_index(&init_x_locs, &init_o_locs);
    let init_state = (init_x_locs, init_o_locs, init_index);
    println!("{:?}", TWO_POWERS);
    println!("state = {:?}", init_state);
    // let init_state = (vec![1, 2, 3], vec![4, 5, 6], 0);
    // state_display(&init_state);
    println!("Hello, world!");
}
