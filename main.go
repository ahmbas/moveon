package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/caneroj1/stemmer"

	"github.com/grokify/html-strip-tags-go"
	"github.com/jbrukh/bayesian"
)

type Ad struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}

func (a Ad) toString() string {
	return toJson(a)
}

func toJson(p interface{}) string {
	bytes, err := json.Marshal(p)
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	return string(bytes)
}

func getAds(path string) []Ad {
	raw, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	var c []Ad
	json.Unmarshal(raw, &c)
	return c
}

const (
	Mover = "Mover"
	Good  = "Good"
)

func main() {

	// Create a classifier with TF-IDF support.
	classifier := bayesian.NewClassifierTfIdf(Mover, Good)
	//movers := getAds("all_movers.json")
	goodAds := getAds("good_sample.json")
	// badStuff := []string{}
	//goodStuff := []string{}

	// for _, a := range movers {
	// 	classifier.Learn(
	// 		strings.Split(a.Title+" "+strip.StripTags(a.Description), " "), Mover)

	// }

	for _, a := range goodAds {
		classifier.Learn(
			strings.Split(a.Title+" "+strip.StripTags(a.Description), " "), Good)

	}

	classifier.ConvertTermsFreqToTfIdf()
	classifier.WriteClassToFile(Good, "/home/ahmed/go/src/github.com/ahmbas/moveon")
	classifier.ReadClassFromFile(Mover, "/home/ahmed/go/src/github.com/ahmbas/moveon")
	sample := getAds("huge_sample.json")
	c := 0
	for _, a := range sample {
		blob := strings.Split(strip.StripTags(a.Title)+" "+strip.StripTags(a.Description), " ")
		probs, likely, _ := classifier.LogScores(blob)
		if likely == 0 {
			fmt.Println(probs, likely, a.Title)
			c++
		}

		//fmt.Println(probs, likely, a.Title)

	}
	//fmt.Println(goodAds)
	fmt.Println(c, len(sample))
	probs, likely, _ := classifier.LogScores(
		stemmer.StemMultiple(strings.Split(
			strip.StripTags("Indoor Flowering plant Easy to care"),
			" ",
		)))
	fmt.Println(probs, likely)

}
