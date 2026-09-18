# MACELES molten-salt potential

`salt-MACELES.model` is the machine-learned interatomic potential used for
every trajectory and derived result under [`Molten_Salts/`](../Molten_Salts/),
including the NaCl system-size benchmark. It is a MACE-family model with long-range
electrostatics (LES), trained on molten chloride salts.

## Identity

| Field | Value |
|---|---|
| File | `salt-MACELES.model` |
| SHA-256 | `15dc639aae415aaf9f6a1e7b5f853836111313b4214dbdafcc1ad9553041a70b` |

The repository does not contain its original source/download metadata. The
checkpoint uses Python/PyTorch serialization and must therefore be treated as
trusted code: the hash checks identity, not authenticity or safety.
