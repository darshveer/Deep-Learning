# MIT 6.875:    Cryptography and Cryptanalysis

- All the topics of cryptography are unified with the fact that there is always an adversary trying to break into the system.
- Say Alice and Bob want to send encrypted messages to each other. They can encrypt the messages absed on some secret key, $S$, such that the the key parameter is the length, $K$, of the key, given by:

$$K = \log_2(S) = |S|$$
- Thus, the cipher key, $c$, when enocded with the encryption algorithm, $E$, on the plaintext message $m$, is:

$$c = E(S, m)$$
- An encryption scheme $(G, E, D)$ is a triplet of (possibly probabilistic) algorithms where:
    - **Key Generation** $G(1^k)$ outputs **secret key** $sk$ of length $k$.
    - **Encryption** $E(sk, m)$ outputs **ciphertext** `c`.
    - **Decryption** $D(sk, c)$ outputs **plaintext** `m`.
- **Correctness**: $D(sk, E(sk, m)) = m \ \forall m \in M$, where $M$ is called the message space, consisting of all possible messages, which have their respective probabilities (needn't be equal).
- Similarly, we have the key space, $K$, and the ciphertext space, $C$. 
- All the algorithms given above are probabilistic, because if it was deterministic, then the output of all of them will always be the same, or will follow a pattern.

## Shannon '49: Perfect Secrecy Theory
- **Adversary**: An adversary is unbounded computationally, and has exponential space and time. We want to create a system that is secure even to such a system.
- **Kercoff Law**: States that the adversary always knows which system is used. What he doesn't know are the secret keys produced.
- We want it to be impossible to:
    - Compute plaintext from ciphertext.
    - Compute an partial information about the plaintext from the ciphertext.
    - Compute relations between plaintexts.
- **Perfect Secrecy**: We say $(G, E, D)$ saisfies **shannon-secrecy** if 
    - $\forall$ probability distribtion over $M$,
        - $\forall \ c \in C$ and $\forall \ m \in M$,
        - $\Pr\limits_ M[\mathcal{M} = m] = \Pr\limits_{M, sk}[\mathcal{M} = m|\mathcal{C} = c]$ where $\mathcal{C} = E(sk, m)$ and $\Pr[C = c] > 0$.
    - This basically means that even if the adversary has the cipher text, he can't know the plaintext. The change in the probability of choosing a cipertext does not change the probability of getting the correct plaintext, or, the obtaining of the ciphertext does not change the message.\
    - But thhis requires that $\bold{|K| \ge |M|}$.
- **Perfect Indistinguishability**: We say $(G, E, D)$ satisfies **perfect indistinguishability** if
    - $\forall$ probability distribtion over $M$,
        - $\forall \ m, m' \in M$,
        - $\forall \ c \in C$,
        - $\Pr\limits_{sk}[\mathcal{C} = c|\mathcal{M} = m] = \Pr\limits_{sk}[\mathcal{C} = c|\mathcal{M} = m']$.
    - $\Pr\limits_{sk}[\mathcal{C} = c|\mathcal{M} = m]$ is the probability, over the randomness of the secret key $sk$, that the encryption of message $m$ produces ciphertext $c$.
    - This states that the probability of obtaining a specific ciphertext $c$ is the same whether the underlying plaintext is $m$ or $m'$, for any pair of messages $m, m' \in M$ and any ciphertext $c \in C$, regardless of the probability distribution over the messages.
    - In simpler terms, the ciphertext $c$ gives no information about whether the plaintext was $m$ or $m'$ (i.e., even if we had the priori probability). An attacker observing $c$ cannot distinguish whether it was generated from $m$ or $m'$, because the probability distribution of ciphertexts is identical for all possible plaintexts.
- Both these defintions are equivalent, that is any scheme that follows perfect secrecy also follows perfect indistinguishability, and vice versa. \
**Proof**: [Proof for Perfect Secrecy $\leftrightarrow$ Indistinguishability](<Proof for Perfect Secrecy, Indistinguishability.pdf>)

### Example of Such a System
The one-time pad (OTP) is the classic example of a cipher that satisfies perfect secrecy. Let’s illustrate how it works:

**Setup**:
- Message space $M = \{0, 1\}^n$ (bitstrings of length $n$).
- Key space $K = \{0, 1\}^n$ (random keys of length $n$).
- Ciphertext space $C = \{0, 1\}^n$.
- Encryption: $E(k, m) = m \oplus k$, where $\oplus$ is bitwise XOR.
- Decryption: $D(k, c) = c \oplus k = (m \oplus k) \oplus k = m$. 

**Key Property:** \
The key $k$ is chosen uniformly at random from $\{0, 1\}^n$, and it is used only once (hence "one-time").
For any plaintext $m \in \{0, 1\}^n$ and any ciphertext $c \in \{0, 1\}^n$, the probability that $c = m \oplus k$ is computed over the random choice of $k$.


**Applying Perfect Indistinguishability:** \
Fix a ciphertext $c$ and two plaintexts $m$ and $m'$.
We need to check $\Pr\limits_{k}[\mathcal{C} = c | \mathcal{M} = m] = \Pr\limits_{k}[\mathcal{C} = c | \mathcal{M} = m']$.
For plaintext $m$, the ciphertext $c = m \oplus k$, so we need $k = c \oplus m$. Since $k$ is chosen uniformly from $\{0, 1\}^n$, the probability that $k = c \oplus m$ is $1/2^n$, as there is exactly one key that satisfies this.
Similarly, for plaintext $m'$, we need $k = c \oplus m'$, and the probability is also $1/2^n$.
Thus, $\Pr\limits_{k}[\mathcal{C} = c | \mathcal{M} = m] = 1/2^n = \Pr\limits_{k}[\mathcal{C} = c | \mathcal{M} = m']$.

## Modern Cryptography
-  A probabilistic polynomial-time algorithm of the order $\mathcal{O}(k^c)$ where for some $c>0$, $k$ is the security parameter. This considers that the adversaries are computationally bounded.

### [Probabilistic Polynomial Time Algorithm (PPT)](PPTs.pdf)
- A probabilistic polynomial-time (PPT) algorithm is a randomized algorithm whose running time is bounded by a polynomial function of the input size $ n $, and it uses random bits (via a random number generator) to make decisions during computation. 
- Unlike deterministic algorithms, which produce the same output for a given input every time, PPT algorithms may produce different outputs for the same input due to randomness, but they are designed to produce correct or useful results with high probability.

**Key Characteristics:**
- *Polynomial Time*: The algorithm’s running time is $ \mathcal{O}(n^k) $, where $ n $ is the input size and $ k $ is a constant (e.g., $ n^2 $, $ n^3 $). This ensures efficiency, as the running time grows polynomially, not exponentially.
- *Randomness*: The algorithm has access to a source of random bits, often modeled as a random tape or a random number generator. This randomness influences decisions, such as choosing a random pivot or sampling a random key.
- *Probabilistic Output*: The algorithm’s output may vary across runs, even for the same input. The correctness or success of the output is typically guaranteed with some probability (e.g., at least $ 1 - \epsilon $, where $ \epsilon $ is a negligible error probability).

### Conventions
- We say that the function $ \epsilon(k) $ is **negligible** if for every polynomial $ P \ \exists \ k_0 \ $ s.t. $ \forall \ k > k_0, \ \epsilon(k) < \frac{1}{P(k)} $.
- We say that the function $ \epsilon(k) $ is **non-negligible** if $ \exists  \ P $, which is a polynommial s.t. $ \forall \ k, \ \epsilon(k) > \frac{1}{P(k)} $.
- We now define the definition of complete secrecy as computational security where the adversaries are computationally *bounded*.

### Computational Indistinguishability
- We say the encryption scheme $ (G,E,D) $ satisfies computational indsinguishability if:
    - $ \forall $ polynomial time samplable message space $ M $,
    - $ \forall $ PPT algorithms EVE,
    - $ \forall $ non-negligible functions $ \epsilon \ \exists \ k_0 $ s.t. $ \forall \ k > k_0 $,
    $$ \Pr[\text{EVE} (m_0, m, c) = b] < \frac{1}{2} + \epsilon(k) $$
    - over $ sk \in_R  G; \ m_1, m_0 \in_R M(1^k); \ b \in_R \{0,1\}; c \in_R E(sk, m_b) $ where "$ \in_R $" signifies randomly chosen from the set. The formula above means the probavility with which EVE can decipher the ciphertext correctly to  be $ m_b $ is less than $ 1/2 + \epsilon(k) $, given that $ m_b $ is either $ m_0 $ or $ m_1 $ and encoded to a length $ b $.
- In PPT time, we can't find $m_0$ and $m_1$ whose ciphertexts ca be told apart for more than a ngligible fraction of time.
- In perfect indistinguishability, it is true that for nay two messages, that every ciphertext occurs with equal probability over the keys.

### Random Encryption
- We introduce a parameter $r$ in encryption, called the randomness parameter or random nonce, which randomizes the input to make the diribution probabilistic.
- In modern cryptohraphy, we cannot assume perfect indisinguishability like the OTP, which got its randomness from the system of generation of keys itself.
- Thus, $r$ is used in practical ciphers to introduce randomness to messages so that the same input message does not always produce the same ciphertext, as every encryption of $m$ with $s$ produces a fresh random value.

### Semantic Security
- We say $(G,E,D)$ satisfies semantic security if:
    - $ \forall\  \text{PPT EVE}$ and $ \forall \ \text{PPT EVE'}$,
    - $ \forall $ polynomial time sampleable distributions $M$,
    - $ \forall $ functions $ f : M \rightarrow \{0,1\} $,
    - $ \forall $ non-negligible functions $ \epsilon > 0,\ \exists \ k_0 $ s.t. $ \forall \ k > k_0 $,
    $$ \left|{\Pr_M\left[\text{EVE}'(1^k) = f(M)\right] - \Pr_{sk \in G(1^k), M}\left[\text{EVE}(1^k,c) = f(M) \ |\  E(sk, M) = c\right]}\right| < \epsilon(k)$$
- We can thus prove that computational indistinguishability follows semantic security and vice versa. 
- **Necessary and sufficient condition:** If one-way funcions exist, then we can have computationally indistinguishable encryption schemes.

## One-Way Functions (OWF)
- A function $f:\{0,1\}^* \rightarrow \{0,1\}^*$ (from all strings to all strings) is a one-way function if:
    1. **Easy-to-evaluate:** $\exists$ PPT algorithm $A$ s.t. $A(x)=f(x) \ \forall \ x$.
    2. **Hard-to-invert:** $\forall$ PPT algorithm Inverter, $\forall$ non-negligible $\epsilon()$, $\exists$ $k_0$ s.t. $ \forall \ k > k_0 $, $ \Pr[\text{Inverter}(1^k, f(x)) = x' \ \ \text{s.t.} \ \ f(x) = f(x')] < \epsilon(k) $, (over $ x \in_R \{0,1\}^k $ and coins of the Inverter.)
- One such hard porblem is computing the factors of any given number. This means that if we take two numbers and multiply them, then *generally* it is hard to recover the factors from that number, as it can have multiple right answers. This works especially good if the factors chosen are primes.