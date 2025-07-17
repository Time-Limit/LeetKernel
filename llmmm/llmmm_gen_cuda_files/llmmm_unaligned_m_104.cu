#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_104_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<104> mm_instantiator;

public:
  UnalignedM_104_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_104_MMInstantiatorWrapper__;

}  // namespace LLMMM
