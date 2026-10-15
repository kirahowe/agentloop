package strictjson

import (
	"encoding/json"
	"reflect"
	"testing"

	"pgregory.net/rapid"
)

type inner struct {
	Class  string   `json:"class"`
	Labels []string `json:"labels,omitempty"`
}

type outer struct {
	Name   string           `json:"name"`
	Tools  map[string]inner `json:"tools"`
	List   []inner          `json:"list"`
	Raw    json.RawMessage  `json:"raw"`
	Ptr    *inner           `json:"ptr"`
	hidden int
}

func TestCheck(t *testing.T) {
	typ := reflect.TypeFor[outer]()
	good := []string{
		`{}`, `{"name":"x"}`,
		`{"tools":{"a":{"class":"read"},"A":{"class":"read","labels":[]}}}`,
		`{"list":[{"class":"x"},{"labels":["p"]}],"raw":{"Any":1,"any":2},"ptr":null}`,
		`{"ptr":{"class":"c"}}`, `[1,2]`, `"s"`,
	}
	for _, g := range good {
		if err := Check([]byte(g), typ); err != nil {
			t.Errorf("%s: %v", g, err)
		}
	}
	bad := []string{
		`{"name":"x","name":"y"}`,
		`{"Name":"x"}`,
		`{"NAME":"x"}`,
		`{"tools":{"a":{"class":"read"},"a":{"class":"act"}}}`,
		`{"tools":{"a":{"Class":"read"}}}`,
		`{"list":[{"labels":["p"],"labels":[]}]}`,
		`{"ptr":{"labels":[],"Labels":[]}}`,
		`{"raw":{"a":1,"a":2}}`,
		`{"hidden":1}`,
	}
	for _, b := range bad {
		if err := Check([]byte(b), typ); err == nil {
			t.Errorf("%s passed", b)
		}
	}
}

// Whatever the keys, an object that passes Check with no type names no
// key twice, and one that names a key twice fails.
func TestPropertyDuplicateKeys(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		keys := rapid.SliceOfN(rapid.SampledFrom([]string{"a", "A", "b", "labels", "Labels"}), 0, 5).Draw(t, "keys")
		b := []byte("{")
		seen, dup := map[string]bool{}, false
		for i, k := range keys {
			if i > 0 {
				b = append(b, ',')
			}
			kb, _ := json.Marshal(k)
			b = append(append(b, kb...), `:[{"x":1}]`...)
			dup = dup || seen[k]
			seen[k] = true
		}
		b = append(b, '}')
		if err := DuplicateKeys(b); (err != nil) != dup {
			t.Fatalf("%s: %v", b, err)
		}
	})
}
