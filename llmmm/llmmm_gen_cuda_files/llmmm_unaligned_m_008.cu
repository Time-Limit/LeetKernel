#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_8_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<8> mm_instantiator;

public:
  UnalignedM_8_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_8_MMInstantiatorWrapper__;

}  // namespace LLMMM
