# c16 summation: worst case and typical error growth

## What we test
(a) That the worst-case bound `|err| <= gamma_n sum|x_i|` for left-to-right summation is nearly attained by a specific input. (b) That for random data the error grows like `sqrt(n) u` for sequential summation, and much more slowly for `torch.sum` (pairwise / tree reduction).

## How
- (a) Input `[2^24, 1, 1, ..., 1]` summed left to right in fp32 on the GPU, one scalar add at a time. `2^24 + 1` is a tie in fp32 and rounds to even, so the sum stays at `2^24` and the error is exactly `n`. Checked at n = 10 ... 10^5 against `gamma_n sum|x_i|`.
- (b) 200 rows of n = 65536 uniform (0,1) fp32 terms. Sequential sums read off at n = 16 ... 65536 (powers of two) against an fp64 cumulative sum; `torch.sum` on the same prefixes. We fit the log-log slope of the median error.

## What the literature says
- Worst case `gamma_n sum|x_i|` (Higham Ch. 3, see c04).
- Under independent zero-mean rounding errors the growth is about `sqrt(n)` (the informal rule) and rigorously `sqrt(n log n)` with high probability (Higham & Mary 2019, confirmed). That pairwise summation has error growing only like `u log n` is the standard result (Higham Ch. 4); I did not confirm the chapter or statement.
- Expected: (a) error / bound close to 1; (b) sequential slope about 0.5, pairwise nearly flat.

## How we use this later
How the error of a long running sum grows (fp32, measured; relative error of the sum of positive terms):

| n | sequential, median | sequential, max | `torch.sum`, max |
|---|---|---|---|
| 16 | 4.0e-8 | 1.9e-7 | 1.2e-7 |
| 1024 | 3.0e-7 | 1.0e-6 | 1.1e-7 |
| 65536 | 2.2e-6 | 8.5e-6 | 1.1e-7 |

- A loop that accumulates one term at a time loses about `sqrt(n) u` (0.5 slope). A tree/pairwise reduction like `torch.sum` stayed at about 2u for every n up to 65536. So for long reductions prefer the library reduction, or accumulate in fp64.
- The worst case is real: `[2^24, 1, ..., 1]` loses every one of the `n` ones, so the error is `n` (absolute), matching `gamma_n sum|x_i|` to 98.8% to 100%. This happens whenever a very large term is accumulated before many small ones.
- For the project: if a quantity is accumulated in a sequential loop over `n` steps (for example a running mean over training steps), expect relative noise `sqrt(n) u`, e.g. 1.9e-5 at n = 10^5 in fp32.
