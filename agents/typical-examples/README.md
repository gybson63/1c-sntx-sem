# typical-examples

Профиль kbshff для `python -m sntx_sem mine-examples`.

Майнер кладёт текст процедуры в `.kuibysheff/mine-input` и читает ответ из `homes/mine/<id>/out/example.json`.

```powershell
mkdir .kuibysheff\mine-input
kbshff config --project-root . --agent typical-examples import --from agents/typical-examples --force
```

В `config.yaml`:

```yaml
examples_miner:
  binary: kbshff
  project_root: "."
  agent: typical-examples
  max_cost: "USD:0.05"
```

`project_root` — каталог, в котором лежит `.kuibysheff` после импорта.
