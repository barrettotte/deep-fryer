# deep-fryer

An image "deep fryer" to learn a little CUDA and parallel programming

<div>
  <a href="examples/moai.jpg">
    <img src="examples/moai.jpg" alt="moai" width="40%" height="40%"/>
  </a>
  <a href="examples/moai-fried.jpg">
    <img src="examples/moai-fried.jpg" alt="moai fried" width="40%" height="40%"/>
  </a>
</div>

<div>
  <a href="examples/cat.png">
    <img src="examples/cat.png" alt="cat" width="40%" height="40%"/>
  </a>
  <a href="examples/cat-fried.jpg">
    <img src="examples/cat-fried.jpg" alt="cat fried" width="40%" height="40%"/>
  </a>
</div>

<div>
  <a href="examples/yuru-camp.jpeg">
    <img src="examples/yuru-camp.jpeg" alt="yuru camp" width="40%" height="40%"/>
  </a>
  <a href="examples/yuru-camp-fried.jpg">
    <img src="examples/yuru-camp-fried.jpg" alt="yuru camp fried" width="40%" height="40%"/>
  </a>
</div>

Examples in [examples/](examples)

There's definitely a lot of optimization I could do, but this is all I'm going to do for now. I hope I revisit CUDA in the near future for something a little more interesting.

## Usage

```sh
./deep-fryer
# Usage: deep-fryer <IMAGE_PATH> [OUTPUT_PATH]

./deep-fryer examples/cat.png
# outputs fried image to ./out.jpg

./deep-fryer examples/moai.jpg examples/moai-fried.jpg
```

## Dependencies

- CUDA
- ImageMagick (for CImg)

## References

- https://deepfriedmemes.com/
