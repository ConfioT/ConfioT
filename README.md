# Confiot

ConfioT-Verifier automatically build ConfiGraph models that specify configuration capabilities implemented by real IoT vendors, and comprehensively reason about configuration risks in real IoT devices based on automated model checking.



# USAGE



### Confiot-verifier

```bash
cd ConfioT/
spin -a IoTConfiguration.pml
gcc -DMEMLIM=16384 -DVECTORSZ=16380 -O2 -DXUSAFE -DSAFETY -DNOCLAIM -DBITSTATE -o pan pan.c
./pan -m2000 -E -e -n > result/result.txt
ls *.trail | xargs -I {} sh -c "spin -k {} -t IoTConfiguration.pml > result/{}.txt"
```



### Large-scale Assessment on Delegation of Smart Devices in MiHome

> All models and analysis results can be found at  https://github.com/ConfioT/ConfioT/tree/main/src/Confiot_main/VIG-parser/react-parser/javascript/MihomePlugins



```
cd src/Confiot_main/VIG-parser/react-parser
python3 parser.py
```





# Role variables in MiHome

1. "sdk_api_level":10025

```
      if (_miot.Service.account.ID == _miot.Device.owner.ID) {
        menuList.splice(1, 0, {
          name: _Localized.Localized.Setting_Title,
          func: function func() {
            return _this.props.navigator.navigate('Setting', {});
          }
        });
      }
```



2. Role variables

```
# https://github.com/MiEcosystem/miot-plugin-sdk/wiki/02-%E5%9F%BA%E7%A1%80%E6%8F%92%E4%BB%B6%E8%AE%BE%E5%A4%87%E6%A8%A1%E5%9D%97
Device.isOwner ⇒ boolean
Device.isFamily ⇒ boolean
Device.isShared ⇒ boolean
Device.isBinded ⇒ boolean
Device.isReadonlyShared ⇒ boolean
```