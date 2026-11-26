skill "extract" {
  policy: "Write out/example.json only. Quote procedure code and do not invent calls."
  allowed_tools: [
    "home.list",
    "home.read",
    "home.write"
  ]
}
