// Local research harness for Mouchet, Bertrand & Hubaux's Thresholdize/Combine.
// Parties share an address space. This is NOT a distributed security boundary.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type Request struct {
	Contributions [][]float64 `json:"contributions"`
	Authorities   int         `json:"authorities"`
	Threshold     int         `json:"threshold"`
	Active        []int       `json:"active"`
	Scale         float64     `json:"scale"`
	Workers       int         `json:"workers"`
}
type Result struct {
	Values               []float64          `json:"values"`
	Timing               map[string]float64 `json:"timing"`
	Ciphertexts          int                `json:"ciphertexts"`
	CiphertextBytes      int                `json:"ciphertext_bytes"`
	DecryptionShareBytes int                `json:"decryption_share_bytes"`
	Slots                int                `json:"slots"`
	PlaintextModulus     uint64             `json:"plaintext_modulus"`
	MaxQuantizationError float64            `json:"max_quantization_error"`
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
func seconds(t time.Time) float64 { return time.Since(t).Seconds() }

func run(r Request) Result {
	if r.Authorities < 2 || r.Authorities > 16 || r.Threshold < 2 || r.Threshold > r.Authorities {
		panic("invalid authority threshold")
	}
	if len(r.Active) != r.Threshold {
		panic("commit exactly threshold distinct active authorities")
	}
	seen := map[int]bool{}
	for _, id := range r.Active {
		if id < 1 || id > r.Authorities || seen[id] {
			panic("invalid or duplicate active identity")
		}
		seen[id] = true
	}
	if len(r.Contributions) < 2 || len(r.Contributions) > 10000 || len(r.Contributions[0]) == 0 {
		panic("invalid contributions")
	}
	if r.Scale <= 0 || math.IsNaN(r.Scale) || math.IsInf(r.Scale, 0) || r.Workers < 1 || r.Workers > 32 {
		panic("invalid scale/workers")
	}
	width := len(r.Contributions[0])
	if width > 100000 {
		panic("vector too wide")
	}
	// Addition-only parameters; no multiplication/relinearization keys required.
	// This research parameter set has NOT received an independent security audit.
	primeGen := ring.NewNTTFriendlyPrimesGenerator(40, 32768)
	plainMod, err := primeGen.NextUpstreamPrime()
	check(err)
	params, err := bgv.NewParametersFromLiteral(bgv.ParametersLiteral{LogN: 14, LogQ: []int{55, 55}, LogP: []int{55}, PlaintextModulus: plainMod})
	check(err)
	absolutes := make([]float64, width)
	for _, row := range r.Contributions {
		if len(row) != width {
			panic("mismatched vector widths")
		}
		for j, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				panic("nonfinite input")
			}
			absolutes[j] += math.Abs(math.Round(v * r.Scale))
		}
	}
	for _, v := range absolutes {
		if v >= float64(plainMod)/4 {
			panic("conservative plaintext overflow guard")
		}
	}
	out := Result{Timing: map[string]float64{}, Slots: params.MaxSlots(), PlaintextModulus: plainMod, MaxQuantizationError: float64(len(r.Contributions)) / (2 * r.Scale)}
	start := time.Now()
	// Each invocation uses fresh keys and one share per fresh ciphertext. No API
	// accepts existing ciphertexts or supports retries under the same secret key.
	keygen := rlwe.NewKeyGenerator(params)
	secrets := make([]*rlwe.SecretKey, r.Authorities)
	points := make([]multiparty.ShamirPublicPoint, r.Authorities)
	thr := multiparty.NewThresholdizer(params)
	shares := make([]multiparty.ShamirSecretShare, r.Authorities)
	for i := range secrets {
		secrets[i] = keygen.GenSecretKeyNew()
		points[i] = multiparty.ShamirPublicPoint(i + 1)
		shares[i] = thr.AllocateThresholdSecretShare()
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crs, err := sampling.NewPRNG()
	check(err)
	crp := ckg.SampleCRP(crs)
	combined := ckg.AllocateShare()
	for i := range secrets {
		publicShare := ckg.AllocateShare()
		ckg.GenShare(secrets[i], crp, &publicShare)
		ckg.AggregateShares(publicShare, combined, &combined)
		poly, err := thr.GenShamirPolynomial(r.Threshold, secrets[i])
		check(err)
		for j := range shares {
			part := thr.AllocateThresholdSecretShare()
			thr.GenShamirSecretShare(points[j], poly, &part)
			check(thr.AggregateShares(shares[j], part, &shares[j]))
		}
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combined, crp, pk)
	// Never sum secret keys or reconstruct a full decryption key.
	secrets = nil
	out.Timing["key_generation_s"] = seconds(start)
	active := make([]multiparty.ShamirPublicPoint, r.Threshold)
	for i, id := range r.Active {
		active[i] = points[id-1]
	}
	start = time.Now()
	additive := make([]*rlwe.SecretKey, r.Threshold)
	for i, id := range r.Active {
		cmb := multiparty.NewCombiner(params, points[id-1], points, r.Threshold)
		additive[i] = rlwe.NewSecretKey(params)
		check(cmb.GenAdditiveShare(active, points[id-1], shares[id-1], additive[i]))
	}
	out.Timing["combine_s"] = seconds(start)
	encoder := bgv.NewEncoder(params)
	eval := bgv.NewEvaluator(params, nil)
	zero := rlwe.NewSecretKey(params)
	// Experimental flooding, NOT a derived 128-bit leakage bound. See README.
	cks, err := multiparty.NewKeySwitchProtocol(params, ring.DiscreteGaussian{Sigma: math.Exp2(40), Bound: 6 * math.Exp2(40)})
	check(err)
	out.Values = make([]float64, 0, width)
	for offset := 0; offset < width; offset += params.MaxSlots() {
		size := min(params.MaxSlots(), width-offset)
		cts := make([]*rlwe.Ciphertext, len(r.Contributions))
		errs := make([]error, len(r.Contributions))
		jobs := make(chan int, len(cts))
		for i := range cts {
			jobs <- i
		}
		close(jobs)
		start = time.Now()
		var wg sync.WaitGroup
		for worker := 0; worker < min(r.Workers, len(cts)); worker++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				enc := rlwe.NewEncryptor(params, pk)
				coder := bgv.NewEncoder(params)
				for i := range jobs {
					ints := make([]int64, size)
					for j := range ints {
						ints[j] = int64(math.Round(r.Contributions[i][offset+j] * r.Scale))
					}
					pt := bgv.NewPlaintext(params, params.MaxLevel())
					if errs[i] = coder.Encode(ints, pt); errs[i] != nil {
						continue
					}
					cts[i], errs[i] = enc.EncryptNew(pt)
				}
			}()
		}
		wg.Wait()
		for _, e := range errs {
			check(e)
		}
		out.Timing["encryption_s"] += seconds(start)
		start = time.Now()
		total := cts[0]
		for i, ct := range cts {
			out.CiphertextBytes += ct.BinarySize()
			out.Ciphertexts++
			if i > 0 {
				check(eval.Add(total, ct, total))
			}
		}
		out.Timing["addition_s"] += seconds(start)
		start = time.Now()
		sumShare := cks.AllocateShare(total.Level())
		for _, sk := range additive {
			part := cks.AllocateShare(total.Level())
			cks.GenShare(sk, zero, total, &part)
			out.DecryptionShareBytes += part.BinarySize()
			check(cks.AggregateShares(sumShare, part, &sumShare))
		}
		cks.KeySwitch(total, sumShare, total)
		// Zero is a PUBLIC key after collective decryption, not a reconstructed key.
		pt := rlwe.NewDecryptor(params, zero).DecryptNew(total)
		decoded := make([]int64, params.MaxSlots())
		check(encoder.Decode(pt, decoded))
		for _, v := range decoded[:size] {
			out.Values = append(out.Values, float64(v)/r.Scale)
		}
		out.Timing["decryption_s"] += seconds(start)
	}
	return out
}
func main() {
	defer func() {
		if e := recover(); e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
	}()
	var r Request
	decoder := json.NewDecoder(os.Stdin)
	decoder.DisallowUnknownFields()
	check(decoder.Decode(&r))
	check(json.NewEncoder(os.Stdout).Encode(run(r)))
}
